# Deployment Guide 🚀

This project is container-ready and can be deployed to any platform effectively.

## 1. Docker Deployment (Local or Server)

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Steps
1.  **Build the Image**:
    ```bash
    docker build -t turbulence-api .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8080:8080 turbulence-api
    ```

3.  **Test**:
    The API will be available at `http://localhost:8080`. You can use the same `curl` commands as before.

---

## 2. Cloud Deployment (Render.com - Free Tier)

Render is the easiest way to deploy this for free.

1.  **Push your code to GitHub** (if not already).
2.  **Sign up/Login to [Render.com](https://render.com/)**.
3.  Click **New +** -> **Web Service**.
4.  Connect your GitHub repository.
5.  **Configure**:
    - **Name**: `turbulence-api` (or any name)
    - **Runtime**: `Docker`
    - **Region**: Closest to you
    - **Instance Type**: Free
6.  Click **Create Web Service**.

Render will automatically build using the `Dockerfile` and deploy your API.

---

## 3. Production Considerations for AWS/GCP

If deploying to AWS ECS, Kubernetes, or Google Cloud Run:
- The container listens on port **8080**.
- Configure health checks at `/health`.
- Set environment variables if needed (though defaults work fine).
