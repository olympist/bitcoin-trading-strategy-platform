"""
Non-interactive deployment script for Bitcoin Trading Strategy app to GCP
"""
import os
import sys
import time
import zipfile
import tempfile
import shutil
import json
import subprocess
from pathlib import Path

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path('.')
DEPLOYMENT_PACKAGE = "bitcoin_strategy_deployment.zip"
ESSENTIAL_FILES = [
    "app.py",
    "requirements.txt",
    ".streamlit/config.toml",
    "utils",
    "strategy",
    "visualization",
    "data",
    "backtesting",
    "results",
    "pages"  # Include pages directory with strategy dashboards
]

def create_deployment_package():
    """Create a deployment package with all necessary files."""
    logger.info("Creating deployment package...")
    
    # Check if package already exists
    if os.path.exists(DEPLOYMENT_PACKAGE):
        logger.info(f"Using existing package: {DEPLOYMENT_PACKAGE}")
        return DEPLOYMENT_PACKAGE
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Creating package in temporary directory: {temp_dir}")
        
        # Copy essential files
        for file_or_dir in ESSENTIAL_FILES:
            src_path = os.path.join(PROJECT_DIR, file_or_dir)
            dst_path = os.path.join(temp_dir, file_or_dir)
            
            if not os.path.exists(src_path):
                logger.warning(f"Skipping non-existent path: {src_path}")
                continue
                
            if os.path.isdir(src_path):
                logger.info(f"Copying directory: {file_or_dir}")
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                logger.info(f"Copying file: {file_or_dir}")
                # Create parent directories if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
        
        # Create setup script
        setup_script = os.path.join(temp_dir, "setup.sh")
        with open(setup_script, 'w') as f:
            f.write("""#!/bin/bash
# Setup script for Bitcoin Trading Strategy app

# Exit on error
set -e

# Colors for better readability
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Function to print a step
print_step() {
  echo -e "\\n${YELLOW}=== $1 ===${NC}"
}

# Function to print success message
print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error message
print_error() {
  echo -e "${RED}✗ $1${NC}"
}

# Installation directory
INSTALL_DIR="/opt/bitcoin-strategy"

print_step "Creating installation directory"
mkdir -p $INSTALL_DIR
cp -r * $INSTALL_DIR/
cp -r .streamlit $INSTALL_DIR/ 2>/dev/null || true

print_step "Installing system dependencies"
apt-get update
apt-get install -y python3-pip python3-venv nginx

print_step "Setting up Python virtual environment"
cd $INSTALL_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

print_step "Creating Streamlit service"
cat > /etc/systemd/system/streamlit.service << EOF
[Unit]
Description=Streamlit Bitcoin Trading Strategy App
After=network.target

[Service]
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/streamlit run app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

print_step "Setting up NGINX"
cat > /etc/nginx/sites-available/streamlit << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

print_step "Starting services"
systemctl daemon-reload
systemctl enable streamlit
systemctl restart streamlit
systemctl restart nginx

print_step "Checking service status"
sleep 5
systemctl status streamlit --no-pager

print_success "Installation complete!"
echo -e "${GREEN}You can access your application at: http://YOUR_VM_IP${NC}"
echo -e "${GREEN}Check application status: python3 $INSTALL_DIR/check_application_status.py${NC}"
""")
        
        # Create application status checker
        status_checker = os.path.join(temp_dir, "check_application_status.py")
        with open(status_checker, 'w') as f:
            f.write("""#!/usr/bin/env python3
import os
import sys
import time
import socket
import subprocess
import requests

def check_url(url, max_retries=3, retry_delay=5):
    \"\"\"
    Check if a URL is accessible and return detailed status information.
    \"\"\"
    result = {
        "accessible": False,
        "status_code": None,
        "content_length": None,
        "response_time": None,
        "error": None
    }
    
    for i in range(max_retries):
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            result["accessible"] = True
            result["status_code"] = response.status_code
            result["content_length"] = len(response.content)
            result["response_time"] = round((end_time - start_time) * 1000, 2)  # in ms
            return result
            
        except requests.RequestException as e:
            result["error"] = str(e)
            print(f"Attempt {i+1}/{max_retries} failed: {e}")
            if i < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    return result

def display_status(result):
    \"\"\"Display detailed status information.\"\"\"
    print("\\n=== Application Status ===")
    
    if result["accessible"]:
        print("✅ Application is accessible")
        print(f"Status code: {result['status_code']}")
        print(f"Content length: {result['content_length']} bytes")
        print(f"Response time: {result['response_time']} ms")
    else:
        print("❌ Application is not accessible")
        print(f"Error: {result['error']}")
    
    print("\\n=== Service Status ===")
    check_service_status("streamlit")
    check_service_status("nginx")
    
    print("\\n=== Port Status ===")
    check_port_status(5000)  # Streamlit
    check_port_status(80)    # NGINX

def check_service_status(service_name):
    \"\"\"Check systemd service status.\"\"\"
    try:
        output = subprocess.check_output(
            ["systemctl", "status", service_name], 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        if "active (running)" in output:
            print(f"✅ {service_name} service is running")
        else:
            print(f"❌ {service_name} service is not running")
            print(f"Status: {output.split('●')[1].split('\\n')[0].strip() if '●' in output else 'Unknown'}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {service_name} service is not running")
        print(f"Error: {e}")

def check_port_status(port):
    \"\"\"Check if a port is in use.\"\"\"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    if result == 0:
        print(f"✅ Port {port} is open")
    else:
        print(f"❌ Port {port} is not open")
    sock.close()

def main():
    \"\"\"Check application status.\"\"\"
    host_ip = socket.gethostbyname(socket.gethostname())
    
    # Check local URL
    local_url = f"http://localhost:5000/"
    print(f"Checking local URL: {local_url}")
    local_result = check_url(local_url)
    display_status(local_result)
    
    # Check public URL
    public_url = f"http://{host_ip}/"
    print(f"\\n=== Public URL Information ===")
    print(f"You can access the application at: {public_url}")
    
    print("\\n=== Troubleshooting Tips ===")
    print("1. If the application is not accessible, restart the services:")
    print("   sudo systemctl restart streamlit nginx")
    print("2. Check logs for errors:")
    print("   sudo journalctl -u streamlit --no-pager -n 50")
    print("3. Ensure ports 5000 and 80 are open in firewall")

if __name__ == "__main__":
    main()
""")
        
        # Create the zip file
        with zipfile.ZipFile(DEPLOYMENT_PACKAGE, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    logger.info(f"Deployment package created: {DEPLOYMENT_PACKAGE}")
    return DEPLOYMENT_PACKAGE

def run_gcloud_command(cmd):
    """Run a gcloud command and handle errors."""
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error: {e}")
        logger.error(e.stderr)
        return None

def ensure_bucket_exists():
    """Ensure the GCS bucket exists, create it if it doesn't."""
    bucket_name = "bitcoin-strategy-deployment"
    
    # Check if bucket exists
    result = run_gcloud_command(["gsutil", "ls", "-b", f"gs://{bucket_name}"])
    if result is None:
        logger.info(f"Creating bucket: {bucket_name}")
        result = run_gcloud_command(["gsutil", "mb", "-l", "us-west1", f"gs://{bucket_name}"])
        if result is None:
            logger.error("Could not create or access GCS bucket.")
            return None
    
    return bucket_name

def upload_to_gcs(source_file):
    """Upload a file to Google Cloud Storage."""
    bucket_name = ensure_bucket_exists()
    if bucket_name is None:
        return None
    
    dest_name = os.path.basename(source_file)
    dest_path = f"gs://{bucket_name}/{dest_name}"
    
    logger.info(f"Uploading {source_file} to {dest_path}")
    result = run_gcloud_command(["gsutil", "cp", source_file, dest_path])
    if result is None:
        return None
    
    return f"{bucket_name}/{dest_name}"

def ensure_vm_exists():
    """Ensure the VM exists, create it if it doesn't."""
    vm_name = "bitcoin-strategy-vm"
    zone = "us-west1-a"
    
    # Check if VM exists
    result = run_gcloud_command([
        "gcloud", "compute", "instances", "describe", 
        vm_name, "--zone", zone, "--format", "json"
    ])
    
    if result is None:
        logger.info(f"VM {vm_name} does not exist. Creating new VM...")
        result = run_gcloud_command([
            "gcloud", "compute", "instances", "create", vm_name,
            "--zone", zone,
            "--machine-type", "e2-medium",
            "--image-family", "debian-11",
            "--image-project", "debian-cloud",
            "--boot-disk-size", "20GB",
            "--tags", "http-server",
            "--metadata", "startup-script-url=gs://cloud-training/gcpnet/ilb/startup.sh"
        ])
        if result is None:
            logger.error(f"Could not create VM {vm_name}")
            return None
    
    return {"name": vm_name, "zone": zone}

def ensure_firewall_rules():
    """Ensure HTTP and HTTPS firewall rules exist."""
    # Check/create HTTP rule
    result = run_gcloud_command([
        "gcloud", "compute", "firewall-rules", "describe", 
        "default-allow-http", "--format", "json"
    ])
    
    if result is None:
        logger.info("Creating HTTP firewall rule")
        result = run_gcloud_command([
            "gcloud", "compute", "firewall-rules", "create", "default-allow-http",
            "--allow", "tcp:80",
            "--target-tags", "http-server"
        ])
    
    # Check/create HTTPS rule
    result = run_gcloud_command([
        "gcloud", "compute", "firewall-rules", "describe", 
        "default-allow-https", "--format", "json"
    ])
    
    if result is None:
        logger.info("Creating HTTPS firewall rule")
        result = run_gcloud_command([
            "gcloud", "compute", "firewall-rules", "create", "default-allow-https",
            "--allow", "tcp:443",
            "--target-tags", "https-server"
        ])

def create_startup_script():
    """Create a startup script for the VM."""
    script_content = """#!/bin/bash
cd /tmp
gsutil cp gs://BUCKET_NAME/PACKAGE_NAME .
mkdir -p /opt/bitcoin-strategy
unzip -o PACKAGE_NAME -d /opt/bitcoin-strategy
cd /opt/bitcoin-strategy
chmod +x setup.sh
bash setup.sh > /tmp/setup.log 2>&1
"""
    
    # Save script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp:
        temp.write(script_content)
        temp_path = temp.name
    
    logger.info(f"Startup script created at {temp_path}")
    return temp_path

def set_vm_startup_script(script_path):
    """Set the VM's startup script."""
    vm = ensure_vm_exists()
    if vm is None:
        return False
    
    # Upload script to GCS
    gcs_path = upload_to_gcs(script_path)
    if gcs_path is None:
        return False
    
    # Set VM metadata
    result = run_gcloud_command([
        "gcloud", "compute", "instances", "add-metadata", vm["name"],
        "--zone", vm["zone"],
        "--metadata", f"startup-script-url=gs://{gcs_path}"
    ])
    
    return result is not None

def restart_vm():
    """Restart the VM to apply the startup script."""
    vm = ensure_vm_exists()
    if vm is None:
        return False
    
    # Reset the VM
    result = run_gcloud_command([
        "gcloud", "compute", "instances", "reset", vm["name"],
        "--zone", vm["zone"]
    ])
    
    return result is not None

def get_vm_external_ip():
    """Get the VM's external IP address."""
    vm = ensure_vm_exists()
    if vm is None:
        return None
    
    result = run_gcloud_command([
        "gcloud", "compute", "instances", "describe", vm["name"],
        "--zone", vm["zone"],
        "--format", "json(networkInterfaces[0].accessConfigs[0].natIP)"
    ])
    
    if result is None:
        return None
    
    try:
        ip_info = json.loads(result)
        return ip_info.get("networkInterfaces")[0].get("accessConfigs")[0].get("natIP")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Error parsing IP address: {e}")
        return None

def main():
    """Main deployment function."""
    logger.info("=" * 80)
    logger.info("🚀 Bitcoin Trading Strategy App - GCP VM Deployment Tool")
    logger.info("=" * 80)
    
    # Create deployment package
    package_path = create_deployment_package()
    if not os.path.exists(package_path):
        logger.error("❌ Deployment failed: Could not create deployment package.")
        return
    
    print("Deployment package created successfully!")
    print("To deploy the application using SSH, run the following command:")
    print("\n  ./deploy_direct.sh VM_IP USERNAME [SSH_KEY_PATH]")
    print("\nExample:")
    print("  ./deploy_direct.sh 34.83.101.32 your_username")
    print("  ./deploy_direct.sh 34.83.101.32 your_username ~/.ssh/id_rsa")
    
    print("\nAlternatively, you can upload the package manually:")
    print(f"1. Upload {package_path} to your VM")
    print("2. Extract it: unzip bitcoin_strategy_deployment.zip")
    print("3. Run the setup script: sudo bash setup.sh")

if __name__ == "__main__":
    main()