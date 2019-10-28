#!/usr/bin/python

from tensorflow.python.client import device_lib

def getAvailableGpu():
	deviceList = device_lib.list_local_devices()
	candidate = None

	for device in deviceList:
		if device.device_type != 'GPU':
			continue
		if candidate == None:
			candidate = device
		else:
			if candidate.memory_limit < device.memory_limit:
				candidate = device

	if candidate != None:
		return candidate.name
	else:
		return None

print ('output_gpu::{}'.format(getAvailableGpu()))
