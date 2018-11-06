import numpy as np

class ModelTrainer:
    """
    Trains a model in rounds as defined by the syllabus with options to pass in
    callbacks to gather information in between.
    """

    def __init__(self, model, syllabus,
        on_task_start=None,
        on_task_complete=None,
        verbose_level=1):
        """
            model:
                A model to train using the passed syllabus

            syllabus:
                A syllabus which handles the distribution of data to be passed to the model
                Must implement the following:
                    training_complete():
                        returns if training should be considered complete
                    next_train():
                        returns the next x and y data to be trained on
                    next_validation():
                        returns the next x and y data to be validated with
                    batch_size():
                        returns the batch_size to be used
                    on_task_complete(history, model):
                        informs the syllabus of a round of training completion

            on_task_start:
                A callback that will be called before a round of training begins
                useful for any preprocessing of data if needed
                function signature -> (data, model, syllabus)
                data = {
                    'x' :  training examples x  , 'y' : training examples y ,
                    'val_x' : validation x      , 'val_y' : validation_y
                }

            on_task_complete:
                A callback that will be called after a round of training has completed
                function signature -> (history, model, syllabus)
                history.history is a dict with scores
                use history.history.keys() to see what you can get
                //Keras give no documentation on the History class
        """
        self.model = model
        self.syllabus = syllabus
        self.verbose_level = verbose_level
        self._on_task_start = on_task_start
        self._on_task_complete = on_task_complete


    def train(self):
        """Trains the model according to the syllabus set"""
        data = dict()
        while not self.syllabus.training_complete():

            #Ask the syllabus for the next set of training and validation samples
            data['x'], data['y'] = self.syllabus.next_train()
            data['val_x'], data['val_y'] = self.syllabus.next_validation()

            if self._on_task_start: #If on_task_start callback is set
                self._on_task_start(data, self.model, self.syllabus)

            #Train the model
            history = self.model.fit(
                x=data['x'],
                y=data['y'],
                epochs=1,
                verbose=self.verbose_level,
                batch_size=self.syllabus.batch_size(),
                shuffle=False,
                validation_data=(data['val_x'], data['val_y'])
            )

            if self._on_task_complete: #If on_task_complete callback is set
                self._on_task_complete(history, self.model, self.syllabus)

            #Inform the syllabus the current round of training has completed
            self.syllabus.on_task_complete(history, self.model)

    def on_task_start(self, function):
        """Sets the on_task_start callback"""
        self._on_task_start = function

    def on_task_complete(self, function):
        """Sets the on_task_complete callback"""
        self._on_task_complete = function
