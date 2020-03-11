import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run, Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import AzureCliAuthentication

print("*********************************************")
print("INSIDE aml-pipeline-for-build.py")
print("*********************************************")
mlScriptsDir = 'scripts-ml'


print('1. Parse arguments...')
print('.............................................')
parser = argparse.ArgumentParser("aml-pipeline-for-build")
parser.add_argument("--aml_compute_target", type=str, help="compute target name", dest="aml_compute_target", required=True)
parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--build_number", type=str, help="build number", dest="build_number", required=True)
parser.add_argument("--image_name", type=str, help="image name", dest="image_name", required=True)
parser.add_argument("--path", type=str, help="path", dest="path", required=True)
args = parser.parse_args()

print("Azure ML SDK version:", azureml.core.VERSION)
print("Argument 1 (aml compute target): %s" % args.aml_compute_target)
print("Argument 2 (model name): %s" % args.model_name)
print("Argument 3 (build number): %s" % args.build_number)
print("Argument 4 (container image name): %s" % args.image_name)
print("Argument 5 (path): %s" % args.path)
print('..1. completed')
print('')
print('')
print('2.  Authenticate...')
print('.............................................')
cliAuth = AzureCliAuthentication()
print('..2. completed')
print('')
print('')
print('3. Instantiate AML workspace object ref...')
print('.............................................')
amlWs = Workspace.from_config(path=args.path, auth=cliAuth)
amlWsStorageRef = amlWs.get_default_datastore()
print('..3. completed')
print('')
print('')
print('4. Instantiate AML managed compute ref...')
print('.............................................')
amlTrainingComputeRef = AmlCompute(amlWs, args.aml_compute_target)
print('..4. completed')
print('')
print('')

print("5. Instantiate and configure run object for the managed compute...")
print('.............................................')
# Create runconfig object
amlComputeRunConf = RunConfiguration()
# Use the compute provisioned
amlComputeRunConf.target = args.aml_compute_target
# Enable Docker
amlComputeRunConf.environment.docker.enabled = True
# Set Docker base image to the default CPU-based image
amlComputeRunConf.environment.docker.base_image = DEFAULT_CPU_IMAGE
# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
amlComputeRunConf.environment.python.user_managed_dependencies = False
# Auto-prepare the Docker image when used for execution (if it is not already prepared)
amlComputeRunConf.auto_prepare_environment = True
# Specify CondaDependencies obj, add necessary packages
amlComputeRunConf.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'numpy',
    'pandas',
    'scikit-learn',
    'azureml-sdk'
])
print("..5. completed")
print('')
print('')

print("6. Define pipeline stage - training...")
print('.............................................')
training_output = PipelineData('train_output', datastore=amlWsStorageRef)
trainPipelineStep = PythonScriptStep(
    name="train",
    script_name="train.py", 
    arguments=["--model_name", args.model_name,
              "--build_number", args.build_number],
    outputs=[training_output],
    compute_target=amlTrainingComputeRef,
    runconfig=amlComputeRunConf,
    source_directory=mlScriptsDir,
    allow_reuse=False
)
print("..6. completed")
print('')
print('')

print("7. Define pipeline stage - containerize...")
print('.............................................')
containerize_output = PipelineData('containerize_output', datastore=amlWsStorageRef)
containerizePipelineStep = PythonScriptStep(
    name="containerize",
    script_name="containerize.py", 
    arguments=["--model_name", args.model_name,  
               "--image_name", args.image_name, 
               "--output", containerize_output],
    outputs=[containerize_output],
    compute_target=amlTrainingComputeRef,
    runconfig=amlComputeRunConf,
    source_directory=mlScriptsDir,
    allow_reuse=False
)
print("..7. completed")
print('')
print('')

print("8. Define pipeline stages sequence, and pipeline itself...")
print('.............................................')
containerizePipelineStep.run_after(trainPipelineStep)
pipeLineSteps = [containerizePipelineStep]
pipeline = Pipeline(workspace=amlWs, steps=pipeLineSteps)
pipeline.validate()
print("..8. completed")
print('')
print('')

print("9. Create run object for the experiment...")
print('.............................................')
run = Run.get_context()
experimentName = run.experiment.name
print("..9. completed")
print('')
print('')

print("10. Submit build pipeline run, synchronously/blocking...")
print('.............................................')
pipelineRun = Experiment(amlWs, experimentName).submit(pipeline)
pipelineRun.wait_for_completion(show_output=True)
print("..10. completed")
print('')
print('')

print("11. Download pipeline output...")
print('.............................................')
# Get a handle to the output of containerize pipeline stage
pipelineStagesLog = pipelineRun.find_step_run('containerize')[0].get_output_data('containerize_output')
# Download locally
pipelineStagesLog.download('.', show_progress=True)
print("..11. completed")
print('')
print('')

print("12. Parse pipeline stages log into JSON...")
print('.............................................')
import json
# load the pipeline output json
with open(os.path.join('./', pipelineStagesLog.path_on_datastore, 'containerize_info.json')) as f:
    buildPipelineOutputVarsJson = json.load(f)

print(buildPipelineOutputVarsJson)
print("..12. completed")
print('')
print('')

print("13. Persist pipeline stages output json for use by the release pipeline...")
print('.............................................')
buildOutputFileName = "build-pipeline-output-vars.json"
outputDir = os.path.join(args.path, 'outputs')
os.makedirs(outputDir, exist_ok=True)
buildOutputFilePath = os.path.join(outputDir,buildOutputFileName)

with open(buildOutputFilePath, "w") as f:
    json.dump(buildPipelineOutputVarsJson, f)
    print('Output file saved! -', buildOutputFilePath)

print("..13. completed")
print('')
print('')
print('************************************************')
print("EXITING aml-pipeline-for-build.py")
print('************************************************')