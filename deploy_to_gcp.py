#!/usr/bin/env python3
"""
Bitcoin Trading Strategy Backtester - Google Cloud Deployment Script

This script deploys the Bitcoin Trading Strategy Backtester application to 
Google Cloud Platform, either to a Cloud Storage bucket or a Compute Engine VM.
"""

import os
import argparse
import sys
from gcp_deploy import auth
from gcp_deploy.deploy import deploy_to_storage, deploy_to_vm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy Bitcoin Trading Strategy Backtester to Google Cloud"
    )
    
    # Deployment method
    parser.add_argument(
        "--method", 
        choices=["storage", "vm", "both"], 
        default="both",
        help="Deployment method: storage (Cloud Storage only), vm (Compute Engine), both (default)"
    )
    
    # Storage options
    parser.add_argument(
        "--bucket", 
        type=str, 
        help="Google Cloud Storage bucket name (created if not exists)"
    )
    
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="bitcoin-backtester",
        help="Prefix for files in the storage bucket (default: bitcoin-backtester)"
    )
    
    # VM options
    parser.add_argument(
        "--vm-name", 
        type=str, 
        default="bitcoin-backtester-vm",
        help="Name for the Compute Engine VM (default: bitcoin-backtester-vm)"
    )
    
    parser.add_argument(
        "--machine-type", 
        type=str, 
        default="e2-medium",
        help="Machine type for the VM (default: e2-medium)"
    )
    
    parser.add_argument(
        "--zone", 
        type=str, 
        default="us-central1-a",
        help="Zone for the VM (default: us-central1-a)"
    )
    
    return parser.parse_args()

def check_credentials():
    """Verify Google Cloud credentials are properly set up."""
    print("Verifying Google Cloud credentials...")
    
    # Ensure the required environment variables are set
    required_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these environment variables and try again.")
        print("You can set them in Replit Secrets with the same names.")
        return False
    
    # Verify credentials
    if not auth.verify_credentials():
        print("❌ Failed to verify Google Cloud credentials.")
        print("Please check that your credentials are valid and have the necessary permissions.")
        return False
    
    return True

def main():
    """Main deployment function."""
    args = parse_arguments()
    
    print("Bitcoin Trading Strategy Backtester - Google Cloud Deployment")
    print("=" * 70)
    
    # Check credentials
    if not check_credentials():
        sys.exit(1)
    
    # Initialize variables
    storage_result = {"success": False, "bucket_name": None, "app_prefix": None, "error": None}
    vm_result = {"success": False, "vm_external_ip": None, "error": None}
    
    # Perform deployment based on method
    if args.method in ["storage", "both"]:
        print("\n🚀 Deploying to Google Cloud Storage...")
        storage_result = deploy_to_storage(
            bucket_name=args.bucket,
            app_prefix=args.prefix
        )
        
        if not storage_result["success"]:
            error_msg = storage_result.get("error", "Unknown error")
            print(f"❌ Storage deployment failed: {error_msg}")
            if args.method == "both":
                print("Cannot proceed with VM deployment.")
                sys.exit(1)
    
    if args.method in ["vm", "both"]:
        print("\n🚀 Deploying to Google Cloud Compute Engine VM...")
        vm_result = deploy_to_vm(
            vm_name=args.vm_name,
            machine_type=args.machine_type,
            bucket_name=args.bucket if args.method == "vm" else storage_result.get("bucket_name"),
            app_prefix=args.prefix,
            zone=args.zone
        )
        
        if vm_result["success"]:
            print("\n✅ Deployment completed successfully!")
            if vm_result["vm_external_ip"]:
                print(f"\n📱 Access your application at: http://{vm_result['vm_external_ip']}")
                print("Note: It may take a few minutes for the VM to fully initialize.")
        else:
            error_msg = vm_result.get("error", "Unknown error")
            print(f"\n❌ VM deployment failed: {error_msg}")
            sys.exit(1)
    
    # If only storage deployment was successful
    if args.method == "storage" and storage_result["success"]:
        print("\n✅ Storage deployment completed successfully!")
        bucket_name = storage_result.get("bucket_name", "unknown-bucket")
        app_prefix = storage_result.get("app_prefix", "unknown-prefix")
        print(f"Files are available in: gs://{bucket_name}/{app_prefix}/")

if __name__ == "__main__":
    main()