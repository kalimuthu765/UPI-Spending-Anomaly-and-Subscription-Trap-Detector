# 🐳 Azure Docker Deployment Guide (Azure Container Apps & App Service)

This guide shows how to deploy the **UPI Spending Anomaly & Subscription Trap Detector** using **Docker** on Microsoft Azure with your **Azure Student Free Account**.

Both the **React Frontend** and **FastAPI Backend + ML Engine** are packaged into a **single unified production Docker container**. This means you only need to run **one container**, completely eliminating multi-container configuration overhead, CORS issues, and extra cloud hosting costs.

---

## 🏗️ Architecture

```
                                  [ User Browser ]
                                         │
                                   HTTP Port 80
                                         │
                          ┌──────────────▼──────────────┐
                          │   Unified Docker Container  │
                          │                             │
                          │   FastAPI + Starlette       │
                          │   ├── / ➔ React Frontend    │
                          │   ├── /assets/ ➔ Static JS  │
                          │   ├── /api/* ➔ ML & DB API  │
                          │   └── /docs ➔ Swagger API   │
                          │                             │
                          │   SQLite / Models Embedded  │
                          └─────────────────────────────┘
```

---

## 🚀 Method 1: Azure Container Apps (Recommended — 100% Free)

Azure Container Apps has a permanent free tier each month:
- **180,000 vCPU-seconds** free
- **360,000 GiB-seconds** free
- **2 Million requests** free

### Step 1: Open Azure Portal
1. Navigate to [portal.azure.com](https://portal.azure.com).
2. In the top search bar, search for **Container Apps** and click **+ Create** ➔ **Container App**.

### Step 2: Configure Basics
- **Subscription**: Select **Azure for Students**
- **Resource group**: Click **Create new** ➔ `upi-detector-rg`
- **Container app name**: `upi-detector-app`
- **Region**: Choose a region with Container Apps support (e.g. `East US`, `Southeast Asia`, `Central US`)
- **Container Apps Environment**: Click **Create new** (keep default consumption plan)

### Step 3: Configure Container
- Under **Image source**, you can choose:
  - **Quickstart image** (or choose your container registry)
  - Or select **Docker Hub or other registries**:
    - **Image type**: Public
    - **Registry login server**: `ghcr.io`
    - **Image and tag**: `ghcr.io/kalimuthu765/upi-spending-anomaly-and-subscription-trap-detector:latest`
- **CPU and Memory**: Select `0.5 vCPU, 1.0 GiB RAM` (covered under free monthly quota)

### Step 4: Configure Ingress (Web Access)
1. Click the **Ingress** tab:
   - **Ingress**: Select **Enabled**
   - **Ingress traffic**: Select **Accepting traffic from anywhere**
   - **Target port**: Enter `80`
   - **Transport**: `Auto` (or `HTTP/1.1`)
2. Click **Review + create** ➔ **Create**.
3. Once provisioned, click **Go to resource** and copy your **Application URL** (e.g., `https://upi-detector-app.xxxx.azurecontainerapps.io`).
4. Open the URL in your browser to access the live app!

---

## 🌐 Method 2: Azure App Service (Web App for Containers)

If you prefer Azure App Service:
1. In Azure Portal, search for **App Services** and click **+ Create** ➔ **Web App**.
2. **Basics Tab**:
   - **Subscription**: `Azure for Students`
   - **Resource group**: `upi-detector-rg`
   - **Name**: `upi-detector-web` (must be globally unique)
   - **Publish**: Select **Container**
   - **Operating System**: **Linux**
   - **Region**: Same as resource group
   - **Pricing Plan**: Select **Free F1** or **Basic B1** (covered by your $100 student credit)
3. **Container Tab**:
   - **Image Source**: **Other container registries**
   - **Access Type**: **Public**
   - **Registry options**: Enter registry server `ghcr.io` and image `ghcr.io/kalimuthu765/upi-spending-anomaly-and-subscription-trap-detector:latest`
4. **Environment / Port Configuration**:
   - Go to **Configuration** / **Environment variables** after creation.
   - Add app setting: `WEBSITES_PORT` = `80`
5. Save and open your `https://upi-detector-web.azurewebsites.net` URL!

---

## 💻 Testing / Running with Docker Locally

To build and run the unified container locally on your own computer:

```powershell
# From the project root:
docker build -t upi-detector .

# Run the container:
docker run -d -p 80:80 --name upi-app upi-detector

# Open in browser:
http://localhost/
```

To stop the container:
```powershell
docker stop upi-app && docker rm upi-app
```
