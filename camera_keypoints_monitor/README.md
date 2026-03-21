The purpose of this cronjob is to alert if there
is a movement of a camera. We do this by collecting 
all the camera infroamtion from the database, downloading
the last frame of the last video made by the camera, and 
checking if the T intersection on the corut has moved by 
a significant amount. 

This script is autaomtically deployed as a Kubernetes cronjob. 
Please see the yaml file in this folder for deployment details 
(currently called cronjob_camera_moves.yaml). It currently runs 
once a day around 2 am ET. You can deplay this cronjob using 
`kubectl apply -f cronjob_camera_moves.yaml`. This makes it so
the cronjob is ready to run for the next time. You can delete it 
with `kubectl delete cronjob check-cameras-moved`. This will 
launch a kubernetes job that will be cleaned up automatically 
after a few days (everything is specified in the spec). 

Note for testing: if yo uwant to run this locally for testing (dry-run, read-only),
there is a test flag in the camera_keypoints_main.py file (is_test at the bottom). 
This will still read from prod data, but wont write to the DB or slack. 
