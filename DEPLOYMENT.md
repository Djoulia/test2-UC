# Deployment Guide - Vercel

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm i -g vercel`
3. **GitHub Repository**: Your code should be pushed to GitHub

## Environment Variables

Before deploying, you'll need to set up these environment variables in Vercel:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `PARADIGM_API_KEY`: Your LightOn Paradigm API key
- `DEBUG`: Set to "false" for production

## Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Connect Repository**:
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository: `Apples96/scaffold-ai-test2`

2. **Configure Project**:
   - Framework Preset: `Other`
   - Root Directory: `./` (leave as default)
   - Build Command: Leave empty (Vercel will auto-detect)
   - Output Directory: Leave empty

3. **Set Environment Variables**:
   - Go to Project Settings → Environment Variables
   - Add the required API keys mentioned above

4. **Deploy**:
   - Click "Deploy"
   - Vercel will automatically build and deploy your app

### Option 2: Deploy via CLI

1. **Login to Vercel**:
   ```bash
   vercel login
   ```

2. **Deploy**:
   ```bash
   vercel
   ```

3. **Set Environment Variables**:
   ```bash
   vercel env add ANTHROPIC_API_KEY
   vercel env add PARADIGM_API_KEY
   vercel env add DEBUG
   ```

4. **Redeploy with environment variables**:
   ```bash
   vercel --prod
   ```

## Post-Deployment

1. **Test Your API**:
   - Your API will be available at: `https://your-project-name.vercel.app`
   - API docs: `https://your-project-name.vercel.app/docs`

2. **Update Frontend** (if needed):
   - Update the API endpoint in `frontend/index.html` to point to your Vercel URL
   - Deploy the frontend separately or serve it from the same domain

## Important Notes

- **Serverless Limitations**: Vercel has timeout limits (10 seconds for hobby plan, 60 seconds for pro)
- **Cold Starts**: First request might be slower due to serverless cold starts
- **Environment Variables**: Make sure all required API keys are set in Vercel dashboard
- **CORS**: The app is configured to allow Vercel domains

## Troubleshooting

- Check Vercel deployment logs for build errors
- Verify environment variables are set correctly
- Test API endpoints using the `/docs` endpoint
- Monitor function execution time and memory usage 