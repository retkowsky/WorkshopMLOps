import argparse
import os, json, sys

import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.model import Model
import azureml.core
from azureml.core import Run
from azureml.core.webservice import AciWebservice, Webservice

from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage
from azureml.core import Image

print("*********************************************")
print("INSIDE containerize.py")
print("*********************************************")

deployModelBool = False
currentlyTrainedModelInRegistry = None
currentlyTrainedModelFound = False
previouslyDeployedRestServiceFound = False
previouslyDeployedRestService = None
previouslyDeployedModel = None
logInfoOutputJsonName = "containerize_info.json"
containerizeStepLogInfo = {}

      
print('1.  Parse arguments')
print('.............................................')
parser = argparse.ArgumentParser("containerize")

parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--image_name", type=str, help="image name", dest="image_name", required=True)
parser.add_argument("--output", type=str, help="containerize output directory", dest="output", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.image_name)
print("Argument 3: %s" % args.output)

os.makedirs(args.output, exist_ok=True)
containerizeStepFilePath = os.path.join(args.output, logInfoOutputJsonName)
print('..1.completed')
print('')
print('')

print('2.  Get run and workspace reference')
print('.............................................')
run = Run.get_context()
amlWs = run.experiment.workspace
print('..2. completed')
print('')
print('')

print('3.  Get freshly trained model details from model registry')
print('.............................................')
modelList = Model.list(amlWs, name = args.model_name)
currentlyTrainedModelInRegistry = sorted(modelList, reverse=True, key = lambda x: x.created_time)[0]

if currentlyTrainedModelInRegistry != None:
    print('..Freshly trained model found!')
    currentlyTrainedModelFound = True
    currentlyTrainedModelId = currentlyTrainedModelInRegistry.id
    print('Currently trained model id: ', currentlyTrainedModelId)

    currentlyTrainedModelName = currentlyTrainedModelInRegistry.name
    print('Currently trained model name: ', currentlyTrainedModelName)

    currentlyTrainedModelVersion = currentlyTrainedModelInRegistry.version
    print('Currently trained model version: ', currentlyTrainedModelVersion)

    currentlyTrainedModelPath = currentlyTrainedModelInRegistry.get_model_path(currentlyTrainedModelName, _workspace=amlWs)
    print('Currently trained model path: ', currentlyTrainedModelPath)

    currentlyTrainedModelRunId = currentlyTrainedModelInRegistry.tags.get("run_id")
    print('Currently trained model run id: ', currentlyTrainedModelRunId)

    currentlyTrainedModelRunRef = Run(run.experiment, run_id = currentlyTrainedModelRunId)
    currentlyTrainedModelAccuracy = currentlyTrainedModelRunRef.get_metrics().get("acc")
    print('Currently trained model accuracy: ', currentlyTrainedModelAccuracy)

    containerizeStepLogInfo["model_name"] = currentlyTrainedModelName
    containerizeStepLogInfo["model_version"] = currentlyTrainedModelVersion
    containerizeStepLogInfo["model_path"] = currentlyTrainedModelPath
    containerizeStepLogInfo["model_acc"] = currentlyTrainedModelAccuracy
    containerizeStepLogInfo["deploy_model_bool"] = deployModelBool
    containerizeStepLogInfo["image_name"] = args.image_name
    containerizeStepLogInfo["image_id"] = ""


else:
    print('..No freshly trained model found!  This should not have happened!')
    
print('..3. completed')
print('')
print('')

print('4.  Determine if we have a REST service deployed for the experiment previously')
print('.............................................')
restServiceList = Webservice.list(amlWs, model_name = currentlyTrainedModelName)
if(len(restServiceList) > 0):
    previouslyDeployedRestServiceFound = True   
    previouslyDeployedRestService = restServiceList[0]
    print('List of existing REST services for the experiment-')
    print(restServiceList)
else:
    print('No existing REST service instances found for the experiment')
    previouslyDeployedRestServiceFound = False

print('..4. completed')
print('')
print('')

print('5.  Get the previously deployed model from previously deployed REST service, if any')
print('.............................................')
previouslyDeployedModel = None
if previouslyDeployedRestService != None:
    try:
        previouslyDeployedContainerImageId = previouslyDeployedRestService.tags['image_id']
        previouslyDeployedContainerImage = Image(amlWs, id = image_id)
        previouslyDeployedModel = previouslyDeployedContainerImage.models[0]
        print('Found the model of the previously deployed REST service, for the experiment!')
    except:
        print('No previously deployed container image not found for the experiment!')
else:
    deployModelBool = True
    print('No deployed Rest service for model: ', currentlyTrainedModelName)
    

print('..5. completed')
print('')
print('')
#deployModelBool = True

print('6.  Determine if the freshly trained model\'s accuracy exceeds that already deployed, if any')
print('.............................................')
previouslyDeployedModelAccuracy = -1 
if previouslyDeployedModel != None:
    previouslyDeployedModelRunRef = Run(run.experiment, run_id = previouslyDeployedModel.tags.get("run_id"))
    previouslyDeployedModelAccuracy = previouslyDeployedModel.get_metrics().get("acc")
    print('Accuracies->freshly trained model versus previously trained and deployed')
    print(currentlyTrainedModelAccuracy, previouslyDeployedModelAccuracy)
    if currentlyTrainedModelAccuracy > previouslyDeployedModelAccuracy:
        deployModelBool = True
        print('Freshly trained model performs better than the model trained and deployed previously!')
    else:
        print('Freshly trained model does NOT perform better than that already deployed!  Aborting deploying freshly trained model!')
        

if deployModelBool == False:
    print('Freshly trained model did not meet the accuracy criteria and will not be deployed!  Persisting output and exiting execution!')
    with open(containerizeStepFilePath, "w") as f:
        json.dump(containerizeStepLogInfo, f)
        print(containerizeStepFilePath, ' saved')
 
    print('..6. completed')
    print('')
    print('')
    print("*********************************************")
    print("EXITING containerize.py")
    print("*********************************************")
    sys.exit(0)
else:
    print('Freshly trained model qualifies for deploying as REST service!')
    print('..6. completed')
    print('')
    print('')

print('7.  Deploy model, if applicable')
print('.............................................')

print('....7.1. Updating scoring file with the correct model name')
with open('score.py') as f:
    data = f.read()
with open('score_fixed.py', "w") as f:
    f.write(data.replace('MODEL-NAME', args.model_name)) 
    print('score_fixed.py saved')
print('....7.1. completed')
print('')

print('....7.2. Creating conda dependencies file')
condaPackages = ['numpy','scikit-learn']
pipPackages = ['azureml-sdk', 'azureml-monitoring']
chdCondaEnv = CondaDependencies.create(conda_packages=condaPackages, pip_packages=pipPackages)

condaDependenciesYamlFile = 'scoring_dependencies.yml'
with open(condaDependenciesYamlFile, 'w') as f:
    f.write(chdCondaEnv.serialize_to_string())

print('....7.2. completed')
print('')

print('....7.3. Creating and registering container image in Azure Container Registry')
containerImageConf = ContainerImage.image_configuration(execution_script = 'score_fixed.py', 
                                                  runtime = 'python', conda_file = condaDependenciesYamlFile)


containerImageForCHD = Image.create(name=args.image_name, models=[currentlyTrainedModelInRegistry], image_config=containerImageConf, workspace=amlWs)

# Block till completion of image creation and registry entry
containerImageForCHD.wait_for_creation(show_output=True)
containerizeStepLogInfo["image_id"] = containerImageForCHD.id

print('....7.3. completed')
print('')

print('....7.4. Persisting output to JSON file')
containerizeStepLogInfo["deploy_model_bool"] = deployModelBool
with open(containerizeStepFilePath, "w") as f:
    json.dump(containerizeStepLogInfo, f)
    print(logInfoOutputJsonName, ' saved')
print('....7.4. completed')
print('')

print('..7. completed')

print("*********************************************")
print("EXITING containerize.py")
print("*********************************************")




