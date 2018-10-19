class Datalist:

    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels
        assert (len(examples) == len(labels))

    def get(self, index):
        return self.examples[index], self.labels[index]

    def get_label(self, index):
        return self.labels[index]

    def get_example(self, index):
        return self.examples[index]

    def get_examples(self):
        return self.examples

    def get_labels(self):
        return self.labels
