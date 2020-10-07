import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import AzureCliAuthentication

print("*********************************************")
print("1. INSIDE provision-aml-compute.py")
print("*********************************************")

# Check core SDK version number
print("Azure ML SDK version:", azureml.core.VERSION)


print('1. Parse arguments...START')
print('.............................................')
parser = argparse.ArgumentParser("provision-aml-compute")
parser.add_argument("--aml_compute_target", type=str, help="compute target name", dest="aml_compute_target", required=True)
parser.add_argument("--path", type=str, help="path", dest="path", required=True)
args = parser.parse_args()
print("Argument 1: %s" % args.aml_compute_target)
print("Argument 2: %s" % args.path)
print('1. Parse arguments...END')
print('')
print('')

print('2. Authenticating...START')
print('.............................................')
cliAuth = AzureCliAuthentication()
print('2. Authenticating...END')
print('')
print('')

print('3.  Get workspace reference...START')
print('.............................................')
amlWs = Workspace.from_config(path=args.path, auth=cliAuth)
print('3.  Get workspace reference...END')
print('')
print('')

print('4.  Get compute reference or create new...START')
print('.............................................')
try:
    amlCompute = AmlCompute(amlWs, args.aml_compute_target)
    print("....found existing compute target.")
except ComputeTargetException:
    print("....creating new compute target")
    
    amlComputeProvisioningConfig = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D12_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 4)    
    amlCompute = ComputeTarget.create(amlWs, args.aml_compute_target, amlComputeProvisioningConfig)
    amlCompute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=10)
    
print('4.  Get compute reference or create new...END')
print('')
print('')
print("*********************************************")
print("EXITING provision-aml-compute.py..")
print("*********************************************")
