### Gesture recognition tensorflow


Please refer to "ros_modules" folder for ros integration

##### Configuration

To let the scripts work properly, you need to setup `config.py` variable in the `ros_module` folder

```
base_path = os.path.join(os.path.expanduser("~"), 'MyTests', 'sofar_gesture_recognition_tf')
```

In this case the project path is `MyTests/sofar_gesture_recognition_tf`, modify it accordance with your system


##### Files description

 * config.py - configurations parameters
 * file_utils.py - utility methods (load, store, prepare data etc..)
 * modules - it actually contains only the model for the gesture which has three properties; starting index, ending index and an array with the data included in those indexes
 * launcher.py - launching scripts for testing purposes
 * rnn_module.py - contains training and test script that creates a fully operational LSTM neural network
 * services - contains scripts for detecting gestures in online environment
     
    