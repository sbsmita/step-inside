import * as path from "path";
import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as ddb from "aws-cdk-lib/aws-dynamodb";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as wafv2 from "aws-cdk-lib/aws-wafv2";
import * as sfn from "aws-cdk-lib/aws-stepfunctions";
import * as tasks from "aws-cdk-lib/aws-stepfunctions-tasks";

const ARM64 = { cpuArchitecture: ecs.CpuArchitecture.ARM64, operatingSystemFamily: ecs.OperatingSystemFamily.LINUX };

export interface VideoConfig {
  model: string;         // luma.ray-v2:0 or amazon.nova-reel-v1:0/1
  region?: string;       // the model's region (default: this stack's)
  bucket?: string;       // where Bedrock writes clips (default: the Scenes bucket, under clips/)
  concurrency: number;   // jobs at once (the account's quota)
  dailyLimit: number;    // paid clips per day
}

export interface StepInsideProps extends cdk.StackProps {
  /** Video for every moment; omitted = no video (moments stay living paintings, the story starts at ~8 s). */
  video?: VideoConfig;
}

export class StepInsideStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: StepInsideProps) {
    super(scope, id, props);
    const video = props?.video;

    // ---------- storage ----------
    const bucket = new s3.Bucket(this, "Scenes", {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      lifecycleRules: [
        { prefix: "jobs/", expiration: cdk.Duration.days(7) }, // raw capture frames are temporary
        { prefix: "clips/", expiration: cdk.Duration.days(1) }, // Nova Reel's raw output (checked clips are copied)
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    const table = new ddb.Table(this, "SceneIndex", {
      partitionKey: { name: "id", type: ddb.AttributeType.STRING },
      billingMode: ddb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ---------- compute (Fargate on Graviton) ----------
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2, natGateways: 0,
      subnetConfiguration: [{ name: "public", subnetType: ec2.SubnetType.PUBLIC }],
    });
    const cluster = new ecs.Cluster(this, "Cluster", { vpc, containerInsights: true });
    const image = new DockerImageAsset(this, "Image", {
      directory: path.join(__dirname, "..", ".."),
      platform: Platform.LINUX_ARM64,
    });
    const logGroup = new logs.LogGroup(this, "AppLogs", {
      retention: logs.RetentionDays.ONE_MONTH, removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const env = {
      SCENE_BUCKET: bucket.bucketName,
      SCENES_TABLE: table.tableName,
      LLM_PROVIDER: "nova", // switch to "claude" once the Anthropic Marketplace agreement is accepted
      NOVA_MODEL: "amazon.nova-pro-v1:0",
      PLANNER_MODEL: "us.amazon.nova-lite-v1:0", // live stories: the story agent, streamed
      IMAGE_REGION: "us-west-2", // Stable Image Core and SD3.5 Large run on demand there
      CORE_RPM: "9", // the account's per-minute quota for Stable Image Core
      STABILITY_DAILY_LIMIT: "400", // paid image calls per day; past it, pictures come from OpenCV and are labelled so
      REC_NETS: "8", // parallel OCR word recognizers (one per vCPU)
      BEDROCK_MODEL: "anthropic.claude-opus-5-5",
      BEDROCK_FALLBACK_MODEL: "anthropic.claude-opus-4-8",
      FULL_WORLDS_PER_STORY: "6", // moments whose own 360 world is repainted by Stability AI (one at a time)
      ...(video ? {
        // Every painted moment becomes a short film; the story starts once its first moment is filmed.
        VIDEO_MODEL: video.model,
        VIDEO_REGION: video.region ?? this.region,
        VIDEO_BUCKET: video.bucket ?? bucket.bucketName,
        VIDEO_CONCURRENCY: String(video.concurrency),
        VIDEO_DAILY_LIMIT: String(video.dailyLimit),
        ANIMATE_MAX: "0", // 0 = film every painted moment
        PLAY_WHEN: "all", // the story opens once it is completely made (every clip made or given up)
        CLIP_DEADLINE_S: "1500",
      } : { STEP_INSIDE_VIDEO: "off", PLAY_WHEN: "still" }),
    };

    // ---------- reconstruction pipeline (Step Functions) ----------
    const workerTask = new ecs.FargateTaskDefinition(this, "WorkerTask", {
      cpu: 4096, memoryLimitMiB: 16384, ephemeralStorageGiB: 30, runtimePlatform: ARM64,
    });
    const worker = workerTask.addContainer("worker", {
      image: ecs.ContainerImage.fromDockerImageAsset(image),
      environment: env,
      command: ["python", "-m", "server.worker"],
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: "worker", logGroup }),
    });
    const runWorker = (name: string, action: string) => new tasks.EcsRunTask(this, name, {
      integrationPattern: sfn.IntegrationPattern.RUN_JOB,
      cluster, taskDefinition: workerTask, assignPublicIp: true,
      subnets: { subnetType: ec2.SubnetType.PUBLIC },
      launchTarget: new tasks.EcsFargateLaunchTarget({ platformVersion: ecs.FargatePlatformVersion.LATEST }),
      containerOverrides: [{
        containerDefinition: worker,
        command: ["python", "-m", "server.worker", action],
        environment: [{ name: "JOB_ID", value: sfn.JsonPath.stringAt("$.job_id") }],
      }],
      resultPath: sfn.JsonPath.DISCARD,
    });

    const failed = new sfn.Fail(this, "Reconstruction failed");
    const done = new sfn.Succeed(this, "Scene ready");
    const reconstruct = runWorker("Reconstruct 3D (OpenCV SfM + dense fusion)", "reconstruct")
      .addRetry({ errors: ["ECS.AmazonECSException", "States.Timeout"], maxAttempts: 2, interval: cdk.Duration.seconds(10) })
      .addCatch(failed, { resultPath: "$.error" });
    // Narration is optional: if Bedrock or Polly is unavailable the scene is still ready.
    const narrate = runWorker("Narrate (Nova Pro on Bedrock + Polly)", "narrate")
      .addCatch(done, { resultPath: "$.error" });

    const stateMachine = new sfn.StateMachine(this, "Pipeline", {
      definitionBody: sfn.DefinitionBody.fromChainable(reconstruct.next(narrate).next(done)),
      timeout: cdk.Duration.minutes(30),
      tracingEnabled: true,
      logs: {
        destination: new logs.LogGroup(this, "PipelineLogs", {
          retention: logs.RetentionDays.ONE_MONTH, removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
        level: sfn.LogLevel.ALL,
      },
    });

    // ---------- API + web app ----------
    // 8 vCPUs: a live story reads the page with 8 OCR recognizers and estimates depth for its moments in parallel.
    // ---------- secrets ----------
    // The one password judges use to create stories. Created once by hand (see README), never in code.
    const appPassword = secretsmanager.Secret.fromSecretNameV2(this, "AppPassword", "step-inside/app-password");
    // Signs login sessions; generated here, never seen by anyone.
    const sessionKey = new secretsmanager.Secret(this, "SessionKey", {
      generateSecretString: { passwordLength: 48, excludePunctuation: true },
    });
    // CloudFront adds this to every request; the app refuses requests without it (no bypassing the firewall).
    const originVerify = new secretsmanager.Secret(this, "OriginVerify", {
      generateSecretString: { passwordLength: 40, excludePunctuation: true },
    });

    const api = new ecsPatterns.ApplicationLoadBalancedFargateService(this, "Api", {
      cluster, cpu: 8192, memoryLimitMiB: 16384, desiredCount: 1, runtimePlatform: ARM64,
      publicLoadBalancer: true, assignPublicIp: true, taskSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      healthCheckGracePeriod: cdk.Duration.seconds(180),
      idleTimeout: cdk.Duration.seconds(300),
      taskImageOptions: {
        image: ecs.ContainerImage.fromDockerImageAsset(image),
        containerPort: 8000,
        environment: { ...env, STATE_MACHINE_ARN: stateMachine.stateMachineArn },
        secrets: {
          APP_PASSWORD: ecs.Secret.fromSecretsManager(appPassword),
          SESSION_KEY: ecs.Secret.fromSecretsManager(sessionKey),
          ORIGIN_VERIFY: ecs.Secret.fromSecretsManager(originVerify),
        },
        logDriver: ecs.LogDrivers.awsLogs({ streamPrefix: "api", logGroup }),
      },
    });
    api.targetGroup.configureHealthCheck({ path: "/api/health", healthyHttpCodes: "200", interval: cdk.Duration.seconds(30) });

    // ---------- permissions ----------
    for (const role of [api.taskDefinition.taskRole, workerTask.taskRole]) {
      bucket.grantReadWrite(role);
      table.grantReadWriteData(role);
      role.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: ["bedrock-mantle:CreateInference"], // Claude in Amazon Bedrock (Messages API endpoint)
        resources: ["*"],
      }));
      // US cross-region inference profiles may route a call to any of these regions.
      const usRegions = ["us-east-1", "us-east-2", "us-west-2"];
      role.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: ["bedrock:InvokeModel"], // Nova Pro (photo narration), Stability Inpaint (picture-book worlds)
        resources: [
          `arn:aws:bedrock:${this.region}::foundation-model/amazon.nova-pro-v1:0`,
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/us.stability.stable-image-inpaint-v1:0`,
          ...usRegions.map((r) => `arn:aws:bedrock:${r}::foundation-model/stability.stable-image-inpaint-v1:0`),
          // Live stories: Stable Image Core paints each moment and the world; SD3.5 Large takes the overflow.
          "arn:aws:bedrock:us-west-2::foundation-model/stability.stable-image-core-v1:1",
          "arn:aws:bedrock:us-west-2::foundation-model/stability.sd3-5-large-v1:0",
        ],
      }));
      role.addToPrincipalPolicy(new iam.PolicyStatement({
        // Live stories: the story agent (Nova Lite, ConverseStream through the US inference profile).
        actions: ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        resources: [
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/us.amazon.nova-lite-v1:0`,
          ...usRegions.map((r) => `arn:aws:bedrock:${r}::foundation-model/amazon.nova-lite-v1:0`),
        ],
      }));
      role.addToPrincipalPolicy(new iam.PolicyStatement({ actions: ["polly:SynthesizeSpeech"], resources: ["*"] }));
      if (video) {
        // The animator runs asynchronously (StartAsyncInvoke is authorised as InvokeModel) and Bedrock writes each
        // clip to the bucket with this role's permissions.
        const vr = video.region ?? this.region;
        role.addToPrincipalPolicy(new iam.PolicyStatement({
          actions: ["bedrock:InvokeModel"],
          resources: [`arn:aws:bedrock:${vr}::foundation-model/${video.model}`],
        }));
        role.addToPrincipalPolicy(new iam.PolicyStatement({
          actions: ["bedrock:GetAsyncInvoke", "bedrock:ListAsyncInvokes"],
          resources: ["*"],
        }));
        if (video.bucket) {
          role.addToPrincipalPolicy(new iam.PolicyStatement({
            actions: ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"],
            resources: [`arn:aws:s3:::${video.bucket}`, `arn:aws:s3:::${video.bucket}/*`],
          }));
        }
      }
    }
    stateMachine.grantStartExecution(api.taskDefinition.taskRole);

    // ---------- HTTPS edge (WebXR and camera access require a secure origin) ----------
    const origin = new origins.LoadBalancerV2Origin(api.loadBalancer, {
      protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY, readTimeout: cdk.Duration.seconds(60),
      customHeaders: { "x-origin-verify": originVerify.secretValue.unsafeUnwrap() }, // resolved by CloudFormation
    });

    // ---------- firewall (AWS WAF, in front of CloudFront; must live in us-east-1) ----------
    let webAclId: string | undefined;
    if (!cdk.Token.isUnresolved(this.region) && this.region === "us-east-1") {
      const managed = (name: string, priority: number, overrides: string[] = []) => ({
        name, priority, overrideAction: { none: {} },
        statement: { managedRuleGroupStatement: { vendorName: "AWS", name,
          ruleActionOverrides: overrides.map((r) => ({ name: r, actionToUse: { count: {} } })) } },
        visibilityConfig: { cloudWatchMetricsEnabled: true, metricName: name, sampledRequestsEnabled: true },
      });
      const acl = new wafv2.CfnWebACL(this, "Firewall", {
        scope: "CLOUDFRONT", defaultAction: { allow: {} },
        visibilityConfig: { cloudWatchMetricsEnabled: true, metricName: "StepInsideFirewall", sampledRequestsEnabled: true },
        rules: [
          // Floods: more than 1500 requests in 5 minutes from one address is blocked for a while.
          { name: "RateLimit", priority: 0, action: { block: {} },
            statement: { rateBasedStatement: { limit: 1500, aggregateKeyType: "IP" } },
            visibilityConfig: { cloudWatchMetricsEnabled: true, metricName: "RateLimit", sampledRequestsEnabled: true } },
          managed("AWSManagedRulesAmazonIpReputationList", 1),
          managed("AWSManagedRulesKnownBadInputsRuleSet", 2),
          // Common attacks (SQL injection, cross-site scripting...). Page photos and videos are big uploads, so the
          // body-size rule only counts.
          managed("AWSManagedRulesCommonRuleSet", 3, ["SizeRestrictions_BODY", "CrossSiteScripting_BODY"]),
        ],
      });
      webAclId = acl.attrArn;
    }

    const cdn = new cloudfront.Distribution(this, "Cdn", {
      webAclId,
      comment: "Step Inside",
      defaultBehavior: {
        origin,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
        originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
      },
      additionalBehaviors: {
        // Scene and story files never change once written: cache them at the edge.
        ...Object.fromEntries(["/api/scenes/*/files/*", "/api/live/*/files/*"].map((p) => [p, {
          origin,
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
          cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
        }])),
      },
    });

    new cdk.CfnOutput(this, "Url", { value: `https://${cdn.distributionDomainName}` });
    new cdk.CfnOutput(this, "Bucket", { value: bucket.bucketName });
    new cdk.CfnOutput(this, "StateMachine", { value: stateMachine.stateMachineArn });
  }
}
