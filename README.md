# PA2-Wine-Prediction

# Wine Quality Prediction AWS Spark Application

This endeavor revolves around crafting a Python application that leverages the PySpark interface. The application is deployed on an Amazon Web Services (AWS) Elastic MapReduce (EMR) cluster. The main goal is to concurrently train a machine learning model on EC2 instances to forecast wine quality based on publicly accessible data. Following the training phase, the model is applied to make predictions about the quality of wine. The deployment process is streamlined through the use of Docker, facilitating the creation of a container image for the trained machine learning model

**GitHub Repository:**
[https://github.com/sumanth-21/PA2-Wine-Prediction/](https://github.com/sumanth-21/PA2-Wine-Prediction/)

**Docker Hub:**
[https://hub.docker.com/repository/docker/sp2927/winequlpred/general](https://hub.docker.com/repository/docker/sp2927/winequlpred/general)

## Execution Steps:

# Wine Quality Prediction AWS EMR Application - Pa2Winepred

1. **Generate an EMR Key Pair:**
   - Navigate to EC2/Network/Key-pairs.
   - Download the key pair in .pem format.
   - Name the key pair: winequality.pem.

2. **Create an AWS S3 Bucket:**
   - Establish a new S3 bucket named: pa2winequalitybucket.

3. **Configure EMR Cluster:**
   - Access the EMR console and set up a new EMR cluster.
   - Configure cluster parameters, scaling, networking, termination, security, EC2 key pair, and IAM roles.
   - Utilize existing configurations or create new ones for efficiency.

4. **Machine Learning Model Training on Spark Cluster:**
   - Connect to the Master instance using SSH:
     ```
     ssh -i "winequality.pem" ec2-user@ec2-44-201-107-82.compute-1.amazonaws.com
     ```
   - Switch to the root user: `sudo su`
   - Submit the task:
     ```
     spark-submit s3://pa2winequalitybucket/winequalityprediction.py
     ```

5. **Run Machine Learning Model using Docker:**
   - Sign up for a Docker account.
   - Download and install Docker on your local system.
   - Build the Docker image:
     ```
     docker build -t winequlpred .
     ```
   - Push and pull from the Docker Hub repository:
     - Push:
       ```
       docker tag winequlpred sp2927/winequlpred
       docker push sp2927/winequlpred
       ```
     - Pull:
       ```
       docker pull sp2927/winequlpred
       ```

6. **Execute Docker Container:**
   - Organize your test data file in a designated folder, named "dir."
   - Connect this directory with the Docker container and execute the container:
     ```
     docker run -v C:\Pa2Winepred\data\csv winequlpred testdata.csv
     ```


