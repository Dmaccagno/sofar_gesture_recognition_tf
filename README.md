### Gesture recognition tensorflow


Please refer to "ros_modules" folder for ros integration

##### Files description

 * config.py - configurations parameters
 * file_utils.py - utility methods (load, store, prepare data etc..)
 * modules - it actually contains only the model for the gesture which has three properties; starting index, ending index and an array with the data included in those indexes
 * launcher.py - launching scripts for testing purposes
 * rnn_module.py - contains training and test script that creates a fully operational LSTM neural network
 * services - contains scripts for detecting gestures in online environment
     
    