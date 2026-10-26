import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";

export interface VideoStackProps extends cdk.StackProps {
  bucketName: string;
}

/** Where Luma AI Ray2 (Amazon Bedrock, us-west-2 only) writes the clips it makes. Bedrock requires the bucket in
 *  the model's region, so this small stack lives in us-west-2; the app copies each checked clip into its own
 *  storage, so clips here expire after a day. Its name is fixed (account + region) so the main stack, in another
 *  region, can refer to it without cross-region exports. Deployed only with `-c video=luma`. */
export class StepInsideVideoStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: VideoStackProps) {
    super(scope, id, props);
    const bucket = new s3.Bucket(this, "Clips", {
      bucketName: props.bucketName,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      lifecycleRules: [{ expiration: cdk.Duration.days(1), abortIncompleteMultipartUploadAfter: cdk.Duration.days(1) }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    new cdk.CfnOutput(this, "VideoBucket", { value: bucket.bucketName });
  }
}
