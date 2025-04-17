#!/usr/bin/env python

"""This script periodically prints the coordinates of the spectator in the CARLA 
simulator. This can be used to get coordinates of actors, objects, roads, or 
anything else in a map."""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 1


def main():
	client = carla.Client(_HOST_, _PORT_)
	client.set_timeout(2.0)
	world = client.get_world()
	
	# print(help(t))
	# print("(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z))
	

	while(True):
		t = world.get_spectator().get_transform()
		# coordinate_str = "(x,y) = ({},{})".format(t.location.x, t.location.y)
		coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
		rotation_str = "(pitch,yaw,roll) = ({},{},{})".format(t.rotation.pitch, t.rotation.yaw,t.rotation.roll)
		print (coordinate_str)
		print (rotation_str)
		time.sleep(_SLEEP_TIME_)



if __name__ == '__main__':
	main()
