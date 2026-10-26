#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { StepInsideStack, VideoConfig } from "../lib/stack";
import { StepInsideVideoStack } from "../lib/video-stack";

const app = new cdk.App();
const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = process.env.CDK_DEFAULT_REGION ?? "us-east-1";

// Video for every moment: off unless chosen at deploy time.
//   -c video=luma   Luma AI Ray2 (us-west-2; accept its AWS Marketplace agreement first; ~$3.75 a clip, 1 at a time)
//   -c video=nova   Amazon Nova Reel (same region as the app; retired in us-east-1 since 2026, kept for regions
//                   that still serve it)
const choice = app.node.tryGetContext("video") as string | undefined;
let video: VideoConfig | undefined;
let videoStack: StepInsideVideoStack | undefined;
if (choice === "luma") {
  if (!account) throw new Error("-c video=luma needs CDK_DEFAULT_ACCOUNT (the clips bucket name includes it)");
  const bucketName = `step-inside-clips-${account}-us-west-2`;
  videoStack = new StepInsideVideoStack(app, "StepInsideVideo", {
    env: { account, region: "us-west-2" }, bucketName,
    description: "Step Inside: bucket for Luma AI Ray2 video clips (Amazon Bedrock, us-west-2)",
  });
  video = { model: "luma.ray-v2:0", region: "us-west-2", bucket: bucketName, concurrency: 1, dailyLimit: 16 };
} else if (choice === "nova") {
  video = { model: "amazon.nova-reel-v1:1", concurrency: 3, dailyLimit: 100 };
}

const main = new StepInsideStack(app, "StepInside", {
  env: { account, region }, video,
  description: "Step Inside: photo/video -> 3D scene with OpenCV 5 on AWS Graviton (OpenCV AI Competition 2026)",
});
if (videoStack) main.addDependency(videoStack);
