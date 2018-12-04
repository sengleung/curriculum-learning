class ModelTrainer:
    """
    Trains a model in rounds as defined by the syllabus with options to pass in
    callbacks to gather information in between.
    """

    def __init__(self,
        model,
        syllabus,
        on_task_start=None,
        on_task_end=None):
        self.model = model
        self.syllabus = syllabus
        self.samples_seen = 0
        self.tasks_complete = 0
        self.results = None
        self.on_task_start=on_task_start
        self.on_task_end=on_task_end

    def train(self):
        """Trains the model according to the syllabus set"""
        while not self.syllabus.training_complete():
            self._train_next()

    def _train_next(self):

        #Inform the syllabus tasks are about to begin with passed data
        self.syllabus.task_starting()

        #Ask the syllabus for the next set of training and validation samples
        x, y, val_x, val_y = self.syllabus.next()

        #If a callback has been set
        if self.on_task_start:
            self.on_task_start()

        #Train the model
        self.results = self.model.fit(
            x=x,
            y=y,
            epochs=1,
            verbose=1,
            batch_size=self.syllabus.batch_size(),
            shuffle=False,
            validation_data=(val_x, val_y)
        )

        #Update how many samples and tasks we've done
        self.samples_seen += len(x)
        self.tasks_complete += 1

        #Inform the syllabus the current round of training has completed
        self.syllabus.task_finished(self.results, self.model)

        #If a callback has been set
        if self.on_task_end:
            self.on_task_end()
