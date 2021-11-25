# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 14:23:23 2021

@author: EZENNIA CHUKUWUDI
"""

class TrackableObject4:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False