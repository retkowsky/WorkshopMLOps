import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core import Image
from azureml.core.authentication import AzureCliAuthentication
import json
import os, sys

print("*********************************************")
print("INSIDE deploy-rest-service.py")
print("*********************************************")

print("Azure Python SDK version: ", azureml.core.VERSION)

print('1. Opening build output json - build-pipeline-output-vars.json...')
buildOutputFilePath = os.path.join('./outputs', 'build-pipeline-output-vars.json')

try:
    with open(buildOutputFilePath) as f:
        buildOutputJson = json.load(f)
        print('..1.completed')
        print(buildOutputJson)
except:
    print("..1.Cannot open: ", buildOutputFilePath)
    print("Exiting...")
    sys.exit(0)

print('')
print('')

print('2. Loading variables')
model_name = buildOutputJson["model_name"]
model_version = buildOutputJson["model_version"]
model_path = buildOutputJson["model_path"]
model_acc = buildOutputJson["model_acc"]
deploy_model_bool = buildOutputJson["deploy_model_bool"]
image_name = buildOutputJson["image_name"]
image_id = buildOutputJson["image_id"]
print('..2.completed')
print('')
print('')


print('3. Determining if model should be deployed')
if deploy_model_bool == False:
    print('..3.Model metric did not meet the metric threshold criteria and will not be deployed!')
    print("*********************************************")
    print("Exiting deploy-rest-service.py")
    print("*********************************************")
    sys.exit(0)

print('..3.completed')
print('')
print('')


print('4. Parsing arguments')
parser = argparse.ArgumentParser("deploy-rest-service")
parser.add_argument("--service_name", type=str, help="service name", dest="service_name", required=True)
parser.add_argument("--aks_name", type=str, help="aks name", dest="aks_name", required=True)
parser.add_argument("--aks_region", type=str, help="aks region", dest="aks_region", required=True)
parser.add_argument("--description", type=str, help="description", dest="description", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.service_name)
print("Argument 2: %s" % args.aks_name)
print("Argument 3: %s" % args.aks_region)
print("Argument 4: %s" % args.description)
print('..4.completed')
print('')
print('')

print('5. Authenticating with AzureCliAuthentication...')
clientAuthn = AzureCliAuthentication()
print('..5.completed')
print('')
print('')

print('6. Instantiate AML workspace')
amlWs = Workspace.from_config(auth=clientAuthn)
print('..6.completed')
print('')
print('')

print('7. Instantiate image')
containerImage = Image(amlWs, id = image_id)
print(containerImage)
print('..7.completed')
print('')
print('')

print('8. Check for and delete any existing web service instance')

aksName = args.aks_name 
aksRegion = args.aks_region
aksServiceName = args.service_name

print('aksName=', aksName)
print('aksRegion=', aksRegion)
print('aksServiceName=', aksServiceName)

try:
    mlRestService = Webservice(name=aksServiceName, workspace=amlWs)
    print(".... Deleting AKS service {}".format(aksServiceName))
    mlRestService.delete()
except:
    print(".... No existing webservice found: ", aksServiceName)

print('..8.completed')
print('')
print('')

print('9. AKS inference cluster creation')

computeList = amlWs.compute_targets
aksTarget = None
if aksName in computeList:
    aksTarget = computeList[aksName]
    
if aksTarget == None:
    print("..... No AKS inference cluster found. Creating new Aks inference cluster: {} and AKS REST service: {}".format(aksName, aksServiceName))
    provisioningConfig = AksCompute.provisioning_configuration(location=aksRegion)
    # Create the cluster
    aksTarget = ComputeTarget.create(workspace=amlWs, name=aksName, provisioning_configuration=provisioningConfig)
    aksTarget.wait_for_completion(show_output=True)
    print(aksTarget.provisioning_state)
    print(aksTarget.provisioning_errors)

print('..9.completed')
print('')
print('')


print('10. REST service creation on AKS cluster')

# Create the service configuration (using defaults)
aksConfig = AksWebservice.deploy_configuration(description = args.description, 
                                                tags = {'name': aksName, 'image_id': containerImage.id})
aksRestService = Webservice.deploy_from_image(
    workspace=amlWs,
    name=aksServiceName,
    image=containerImage,
    deployment_config=aksConfig,
    deployment_target=aksTarget
)
aksRestService.wait_for_deployment(show_output=True)
print(aksRestService.state)

print('..10.completed')
print('')
print('')

print('11. Create output Json with Rest service details')

api_key, _ = aksRestService.get_keys()
print("....Deployed AKS REST service: {} \nREST service Uri: {} \nREST service API Key: {}".
      format(aksRestService.name, aksRestService.scoring_uri, api_key))

aksRestServiceJson = {}
aksRestServiceJson["aksServiceName"] = aksRestService.name
aksRestServiceJson["aks_service_url"] = aksRestService.scoring_uri
aksRestServiceJson["aks_service_api_key"] = api_key
print("....AKS REST service Info")
print(aksRestServiceJson)

print('..11.completed')
print('')
print('')

print('12. Enable model metrics logging')
aksRestService.update(enable_app_insights=True, collect_model_data=True)
print('..12.completed')
print('')
print('')


print("13. Save aksRestServiceJson.json...")
aksRestServiceFilePath = os.path.join('./outputs', 'aksRestServiceJson.json')
with open(aksRestServiceFilePath, "w") as f:
    json.dump(aksRestServiceJson, f)
print('..13.completed')
print('')
print('')


# TODO: Figure out why I am getting a bad request error to the REST call here; Manual test shows successful deployment;
#print("14. Quick test of REST API on one record")
# Single test data
#testApiCallDataset = {"data":[[61, 1, 150, 103]]}
# Call the REST service to make predictions on the test data
#samplePrediction = aksRestService.run(json.dumps(testApiCallDataset))
#print('..... Test data prediction should be 1 (age=61, prevalentHyp = 1, sysBP = 150, glucose = 103): ', samplePrediction)
#print('..14.completed')
#print('')
#print('')


print("*********************************************")
print("EXITING deploy-rest-service.py")
print("*********************************************")

