class ModelTrainer:
    """
    Trains a model in rounds as defined by the syllabus with options to pass in
    callbacks to gather information in between.
    """

    def __init__(self, model, syllabus,
        verbose_level=1):
        self.model = model
        self.syllabus = syllabus
        self.verbose_level = verbose_level

    def train(self):
        """Trains the model according to the syllabus set"""
        while not self.syllabus.training_complete():
            self._train_next()

    def _train_next(self):
        #Ask the syllabus for the next set of training and validation samples
        x, y, val_x, val_y = self.syllabus.next()

        #Inform the syllabus tasks are about to begin with passed data
        self.syllabus.task_starting(x, y, val_x, val_y)

        #Train the model
        history = self.model.fit(
            x=x,
            y=y,
            epochs=1,
            verbose=self.verbose_level,
            batch_size=self.syllabus.batch_size(),
            shuffle=False,
            validation_data=(val_x, val_y)
        )

        #Inform the syllabus the current round of training has completed
        self.syllabus.task_finished(history, self.model)
