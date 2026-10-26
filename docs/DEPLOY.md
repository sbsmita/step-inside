# Deploying Step Inside to AWS (step by step)

You run every command below in the macOS Terminal, from the project folder. It takes about 45 minutes the first
time, most of it waiting. Nothing here needs you to write code.

```bash
cd ~/earth/hackathons/opencv-hackathon
```

## 1. Install the two tools (once)

```bash
brew install --cask finch     # builds the app's container image (a free Docker alternative from AWS)
finch vm init                 # starts Finch's small virtual machine (a few minutes, once)
```

Node.js is already installed. If `npm` is ever missing: `brew install node@22`.

## 2. Log in to AWS and hand the login to the deploy tool

```bash
aws login
aws sts get-caller-identity --query Account --output text      # should print 931628335592
eval "$(aws configure export-credentials --format env)"        # the deploy tool reads the login from here
```

The last line lasts as long as the login session. If a later step says "no credentials" or "session expired",
run these three lines again.

## 3. Create the judges' password (once)

Pick a long password (for example three random words and a number). Share it only with the judges, privately.
Don't paste it into chats or the repository.

```bash
aws secretsmanager create-secret --region us-east-1 \
  --name step-inside/app-password --secret-string 'PUT-YOUR-PASSWORD-HERE'
```

To change it later: `aws secretsmanager put-secret-value --region us-east-1 --secret-id step-inside/app-password
--secret-string 'NEW-PASSWORD'`, then redeploy (step 6) or restart the service.

## 4. Remove the two test buckets made by hand (once)

The deploy creates the video bucket itself, with the same name as one made during testing, so the old ones must go.
Both hold only test clips.

```bash
aws s3 rb s3://step-inside-clips-931628335592-us-west-2 --force
aws s3 rb s3://step-inside-clips-931628335592 --force
```

## 5. Prepare the AWS account for deployments (once)

```bash
cd infra
npm install
npx cdk bootstrap aws://931628335592/us-east-1 aws://931628335592/us-west-2
```

(The app runs in us-east-1; Luma's video bucket lives in us-west-2, so both regions are prepared.)

## 6. Deploy

```bash
CDK_DOCKER=finch npx cdk deploy --all -c video=luma
```

- It builds the container image, then shows the security changes it is about to make and asks
  `Do you wish to deploy these changes (y/n)?`: type `y` and Enter (it may ask once per stack).
- It takes 20–40 minutes. When it finishes it prints a line like
  `StepInside.Url = https://d1234abcd.cloudfront.net`. **That is the app's public address.**

## 7. Check it

1. Open the address in a browser. You should see the Step Inside home page.
2. Click **🔒 Log in**, enter the password from step 3.
3. Upload a page. The progress screen should climb to 100%, then the door appears: click **✨ Step inside**.
4. On the Meta Quest: open the same address in the Quest Browser (no certificate warning this time), log in,
   create or open a story, tap **Enter VR**.

## 8. Give it to the judges

In the Devpost form (the testing-instructions / notes for judges field), write the address and the password, for
example: "Live app: https://… · password: … (needed only to create a story; anyone can watch existing ones)."

## Costs, and switching it off

| What | Roughly |
|---|---|
| The server (Fargate, 8 vCPU / 16 GB, Graviton) | about $0.32 an hour, about $8 a day |
| Load balancer, firewall, logs, storage | about $1 a day |
| Each story created | a few cents of pictures and voices, plus about $3.75 per filmed moment |

Video is capped at 16 clips a day. Set a billing alarm once in the AWS console (Billing → Budgets → Create budget).

When judging is over, remove everything (this deletes the app and its stored stories):

```bash
cd ~/earth/hackathons/opencv-hackathon/infra
eval "$(aws configure export-credentials --format env)"
npx cdk destroy --all
aws secretsmanager delete-secret --region us-east-1 --secret-id step-inside/app-password --force-delete-without-recovery
```

## Updating after a change

Repeat step 2 (login), then step 6. Only what changed is rebuilt.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `no credentials have been configured` / `session expired` | Repeat step 2 |
| `Cannot connect to the Docker daemon` / finch errors | `finch vm start`, then deploy again |
| Deploy fails on the bucket `step-inside-clips-…` | Step 4 was skipped: run it, then deploy again |
| The site shows 503 for a few minutes after deploying | The server is still starting (it downloads its vision models): wait 3–5 minutes |
| Login never accepted | Check the secret: `aws secretsmanager get-secret-value --region us-east-1 --secret-id step-inside/app-password --query SecretString --output text` |
| Anything else | AWS console → CloudWatch → Log groups → the `StepInside-AppLogs…` group shows the server's messages |

## What protects the deployed app

- **The server code never leaves the server.** Only the web page's files are sent to browsers, and those are
  minified (one unreadable line per file, no comments, no source maps). Browser code can never be fully hidden: a
  browser must be able to run it. Nothing secret is in it.
- **One password** guards everything that costs money (creating stories); watching stays open. Wrong passwords are
  limited to 5 a minute per address; the login is an HttpOnly, SameSite=Strict signed cookie.
- **Only CloudFront can reach the app**: CloudFront adds a secret header the app checks, so the load balancer can't
  be used to skip the firewall.
- **AWS WAF firewall**: blocks floods (over 1500 requests in 5 minutes from one address), known bad addresses,
  and common attack patterns.
- **Security headers** on every response (content security policy, no framing, HTTPS only), and no public API
  documentation pages.
- **Spending caps**: at most 3 stories built at once, 400 pictures and 16 video clips a day.
- **Secrets** (the password, the session key, the CloudFront header) live in AWS Secrets Manager, never in code.

Note on the repository: if the competition requires a public repository, the source code is public there anyway.
The protections above are about the running app.
