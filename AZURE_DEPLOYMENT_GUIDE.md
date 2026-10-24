# ☁️ Azure Deployment Guide (Azure for Students)

This guide walks you step-by-step through deploying the **UPI Spending Anomaly and Subscription Trap Detector** on Microsoft Azure using your **Azure Student Free Account**.

---

## 💡 Overview of Deployment Architecture

- **Hosting Platform**: Azure Virtual Machine (Ubuntu 22.04 LTS, Standard_B2s or Standard_B1s)
- **Container Stack**: Docker & Docker Compose
  - **Frontend**: React + Vite + Nginx (Port 80)
  - **Backend**: FastAPI ML API (Proxied via Nginx on `/api/`, direct on Port 8000)
  - **Database**: PostgreSQL (Containerized) or SQLite (Internal)
  - **ML Model Engine**: Scikit-Learn Isolation Forest & Subscription Trap Detector
  - **MLflow Tracking**: Experiment tracking dashboard (Port 5000)
- **Cost**: **$0.00** (100% covered by your Azure for Students $100 credit / 12-month free services)

---

## 🚀 Quick Step-by-Step Deployment (Azure Portal)

### Step 1: Open Azure Portal
1. Navigate to [portal.azure.com](https://portal.azure.com) in your web browser.
2. Sign in with your Azure Student account.

---

### Step 2: Create a Virtual Machine
1. In the top search bar, type **Virtual machines** and select it.
2. Click **+ Create** ➔ **Azure virtual machine**.

#### **A. Basics Tab**
- **Subscription**: Select **Azure for Students**
- **Resource group**: Click **Create new** ➔ name it `upi-detector-rg` ➔ click **OK**
- **Virtual machine name**: `upi-detector-vm`
- **Region**: Choose a region near you (e.g., `Central India`, `Southeast Asia`, or `East US`)
- **Availability options**: `No infrastructure redundancy required`
- **Security type**: `Standard`
- **Image**: **Ubuntu Server 22.04 LTS - x64 Gen2**
- **Size**: 
  - Select **Standard_B2s** (2 vCPUs, 4 GiB RAM — Recommended for fast ML container building, costs ~$0.04/hr which is well within the $100 student credit)
  - *OR* **Standard_B1s** (1 vCPU, 1 GiB RAM — eligible for 12 months free tier)
- **Administrator account**:
  - Authentication type: Select **Password**
  - Username: `azureuser`
  - Password: Create a strong password (e.g. `UpiStudent2026!`)
- **Inbound port rules**:
  - Public inbound ports: **Allow selected ports**
  - Select inbound ports: Check **SSH (22)** and **HTTP (80)**

#### **B. Advanced Tab (Automatic Zero-Touch Setup)**
1. Click on the **Advanced** tab at the top.
2. Scroll down to **Custom data and cloud-init** (or **Custom data**).
3. Paste the following configuration directly into the box:

```yaml
#cloud-config
package_upgrade: true
packages:
  - git
  - curl
  - docker.io
  - docker-compose-v2

runcmd:
  - fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  - chmod 600 /swapfile
  - mkswap /swapfile
  - swapon /swapfile
  - echo '/swapfile none swap sw 0 0' >> /etc/fstab
  - systemctl enable --now docker
  - git clone https://github.com/kalimuthu765/UPI-Spending-Anomaly-and-Subscription-Trap-Detector.git /opt/upi-app
  - cd /opt/upi-app/work && docker compose up -d --build
```

#### **C. Review + Create**
1. Click **Review + create** at the bottom.
2. Wait for validation to pass, then click **Create**.
3. Azure will take 1–2 minutes to provision the VM.

---

### Step 3: Access Your Live Application!
1. When deployment completes, click **Go to resource**.
2. Under the **Overview** section, look for **Public IP address** (e.g. `20.198.xxx.xxx`).
3. Allow 2–3 minutes after VM launch for Docker to finish building the app.
4. In your browser, open:
   ```text
   http://<YOUR_VM_PUBLIC_IP>/
   ```
5. You will see the **UPI Spending Anomaly & Subscription Trap Detector** React dashboard!

---

## 🛠️ Alternative Method: Azure Portal "Run Command" (No SSH Required)

If you didn't paste the cloud-init script or want to run deployment manually without an SSH client:
1. In the Azure Portal, open your VM (`upi-detector-vm`).
2. On the left sidebar menu, scroll down to **Operations** ➔ click **Run command**.
3. Select **RunShellScript**.
4. Paste this single command into the script window:
   ```bash
   git clone https://github.com/kalimuthu765/UPI-Spending-Anomaly-and-Subscription-Trap-Detector.git /opt/upi-app && bash /opt/upi-app/deploy.sh
   ```
5. Click **Run**. Azure executes the full build and start script directly on the VM.

---

## 🌐 Opening Additional Ports (Optional: Swagger Docs & MLflow)

If you want to view FastAPI Swagger Docs (port 8000) or the MLflow UI (port 5000):
1. On your VM page in Azure Portal, go to **Settings** ➔ **Networking** (or **Network settings**).
2. Click **Add inbound port rule**:
   - **Destination port ranges**: `8000,5000`
   - **Protocol**: `TCP`
   - **Action**: `Allow`
   - **Priority**: `310`
   - **Name**: `Allow-App-Ports`
3. Click **Add**.
4. You can now access:
   - **FastAPI Swagger Docs**: `http://<YOUR_VM_PUBLIC_IP>:8000/docs`
   - **MLflow Tracking UI**: `http://<YOUR_VM_PUBLIC_IP>:5000/`

---

## 🛑 How to Stop or Restart to Save Student Credits

- To stop the VM when not in use: Click **Stop** at the top of the VM page in Azure Portal (Status will change to *Deallocated*, so you won't be charged for compute).
- To resume: Click **Start**. The Docker containers will automatically start up with the system.
