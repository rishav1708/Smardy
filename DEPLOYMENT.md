# Deployment Guide for Smart Document Analyzer

This guide provides multiple options to deploy the Smart Document Analyzer application publicly.

## 🚀 Quick Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `rishav1708/smardy`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Environment Variables** (Optional):
   - In Streamlit Cloud dashboard, go to "App settings"
   - Add secrets in the "Secrets" section:
     ```toml
     OPENAI_API_KEY = "your_openai_api_key_here"
     ```

### Option 2: Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t smardy .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 smardy
   ```

3. **Access the app**: Open http://localhost:8501

### Option 3: Heroku Deployment

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create a Heroku app**:
   ```bash
   heroku create your-app-name
   ```

3. **Add Python buildpack**:
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 4: Railway Deployment

1. **Visit [railway.app](https://railway.app)**
2. **Connect your GitHub repository**
3. **Deploy directly from GitHub**

## 🔧 Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Optional - for enhanced AI features
OPENAI_API_KEY=your_openai_api_key_here

# Application settings
APP_NAME=Smart Document Analyzer
DEBUG=False
```

### Port Configuration

The app runs on port 8501 by default. For cloud deployments, ensure your platform uses the correct port.

## 📝 Post-Deployment Steps

1. **Test all features**:
   - Upload a document
   - Try ML analysis
   - Test Q&A functionality (local models only)
   - Check insights generation

2. **Monitor logs** for any issues

3. **Share your deployed app**:
   - Your Streamlit Cloud URL: `https://your-app-name.streamlit.app`
   - Share with users!

## 🎯 Production Tips

- **Large files**: Consider adding file size limits
- **Performance**: Monitor resource usage
- **Security**: Never commit API keys to Git
- **Updates**: Set up automatic deployments from Git

## 🔗 Useful Links

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Docker Hub](https://hub.docker.com/)
- [Heroku Documentation](https://devcenter.heroku.com/)

---

**Happy Deploying! 🚀**
