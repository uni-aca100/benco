### 4.4 Deployment Instructions

Below is the step-by-step procedure for deploying the entire infrastructure and executing the benchmark tests.

#### **1. Inference Server Setup (FakeShield)**

The first phase involves preparing the containerized environment hosting FakeShield's inference microservices.

##### **Step 1: Clone the FakeShield Repository**

Retrieve the server source code and switch to the dedicated service branch (`serve`):

```bash
git clone https://github.com/uni-aca100/FakeShield.git
cd FakeShield
git checkout serve


```

##### **Step 2: Download FakeShield Model Weights**

Download FakeShield's pre-trained weights from Hugging Face into the `weight/` directory:

```bash
pip install huggingface_hub
huggingface-cli download --resume-download zhipeixu/fakeshield-v1-22b --local-dir weight/


```

##### **Step 3: Download SAM (Segment Anything Model) Weights**

Download the SAM model checkpoint used by the MFLM module for segmentation:

```bash
wget https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth -P weight/


```

Once downloads are complete, the `weight/` directory structure should look like this:

```text
FakeShield/
├── weight/
│   ├── fakeshield-v1-22b/
│   │   ├── DTE-FDM/
│   │   ├── MFLM/
│   │   └── DTG.pth
│   └── sam_vit_h_4b8939.pth


```

##### **Step 4: Launch Microservices**

Build the containers and spin up the services in the background using Docker Compose:

```bash
docker-compose build
docker-compose up -d


```

##### **Step 5: Verify Service Status and Model Loading**

Before sending client requests, check the logs to ensure the `DTE_FDM` module, `MFLM` module, and API Gateway have finished initializing and loading weights into VRAM:

```bash
docker-compose logs -f


```

#### **2. Client Setup and Execution (IMDL-BenCo)**

Once the server infrastructure is up, running, and listening, you can prepare and execute the benchmarking client.

##### **Step 6: Clone the Client Repository (`benco`)**

Clone the modified `IMDL-BenCo` benchmark environment with Proxy and IPC support:

```bash
git clone https://github.com/uni-aca100/benco.git
cd benco


```

##### **Step 7: Build the Client Docker Image**

Build the Docker image containing the isolated runtime environment for the client:

```bash
docker build -t benco .


```

##### **Step 8: Run the Test Script**

Launch from the benco directory the client container, granting GPU access, enabling host networking, and sharing the IPC memory segment:

```bash
docker run --gpus all --network host --ipc=host --rm -it -v $(pwd)/benco:/workspace benco bash /workspace/test_mymodel.sh


```