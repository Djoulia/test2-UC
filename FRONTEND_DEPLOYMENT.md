# Frontend Deployment Guide

Your frontend is now configured to communicate with your Vercel API at `https://scaffold-ai-test2.vercel.app`.

## Option 1: Deploy to Netlify (Recommended - Easiest)

### Step 1: Prepare for Deployment
1. Make sure your code is pushed to GitHub
2. The frontend is already configured to use the Vercel API

### Step 2: Deploy via Netlify Dashboard
1. **Go to [netlify.com](https://netlify.com)** and sign up/login
2. **Click "New site from Git"**
3. **Connect to GitHub** and select your repository: `Apples96/scaffold-ai-test2`
4. **Configure build settings**:
   - Build command: (leave empty)
   - Publish directory: `frontend`
5. **Click "Deploy site"**

### Step 3: Custom Domain (Optional)
- Your site will be available at: `https://random-name.netlify.app`
- You can set a custom domain in Site Settings → Domain management

## Option 2: Deploy to Vercel (Separate Project)

### Step 1: Create New Vercel Project
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import the same GitHub repository: `Apples96/scaffold-ai-test2`

### Step 2: Configure for Static Site
1. **Framework Preset**: `Other`
2. **Root Directory**: `./` (leave as default)
3. **Build Command**: (leave empty)
4. **Output Directory**: `frontend`
5. **Install Command**: (leave empty)

### Step 3: Deploy
- Click "Deploy"
- Your frontend will be at: `https://your-frontend-project.vercel.app`

## Option 3: Deploy to GitHub Pages

### Step 1: Create GitHub Pages Branch
```bash
# Create a new branch for GitHub Pages
git checkout -b gh-pages

# Move frontend files to root
mv frontend/* .
rmdir frontend

# Commit and push
git add .
git commit -m "Setup for GitHub Pages"
git push origin gh-pages
```

### Step 2: Enable GitHub Pages
1. Go to your GitHub repository
2. Settings → Pages
3. Source: Deploy from a branch
4. Branch: `gh-pages`
5. Folder: `/ (root)`
6. Click "Save"

Your site will be at: `https://apples96.github.io/scaffold-ai-test2`

## Option 4: Deploy to Firebase Hosting

### Step 1: Install Firebase CLI
```bash
npm install -g firebase-tools
```

### Step 2: Initialize Firebase
```bash
firebase login
firebase init hosting
```

### Step 3: Configure
- Public directory: `frontend`
- Configure as single-page app: `No`
- Overwrite index.html: `No`

### Step 4: Deploy
```bash
firebase deploy
```

## Testing Your Deployment

After deploying, test these features:

1. **Create Workflow**: Enter a description and create a workflow
2. **Execute Workflow**: Run the generated workflow
3. **File Upload**: Upload a file and ask questions about it
4. **Feedback System**: Provide feedback and regenerate workflows

## Troubleshooting

### CORS Errors
If you get CORS errors, make sure:
- Your frontend domain is in the CORS allowlist (already configured)
- The API_BASE URL is correct in `frontend/index.html`

### API Connection Issues
- Check that your Vercel API is deployed and running
- Verify environment variables are set in Vercel
- Test the API directly at: `https://scaffold-ai-test2.vercel.app/docs`

### Build Errors
- Make sure the `frontend` directory contains all necessary files
- Check that `index.html` is in the correct location
- Verify all JavaScript references are correct

## Environment Variables

Your frontend doesn't need any environment variables - it communicates with your Vercel API which handles all the API keys.

## Security Notes

- API keys are stored securely in Vercel environment variables
- Frontend is static and doesn't contain sensitive information
- CORS is properly configured to allow only trusted domains 