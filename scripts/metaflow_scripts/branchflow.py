from metaflow import FlowSpec, step

class CounterBranchFlow(FlowSpec):
    @step
    def start(self):
        self.creature = 'dog'
        self.count = 0
        self.next(self.add_one, self.add_two)  # this is the branching

    @step
    def add_one(self):
        self.count += 1
        self.next(self.join)  # move to merging step

    @step
    def add_two(self):
        self.count += 2
        self.next(self.join)

    @step
    def join(self, inputs):
        self.count = max(inp.count for inp in inputs)
        print(f"Count from add_one - {inputs.add_one.count}")
        print(f"Count from add_two - {inputs.add_two.count}")

        self.creature = inputs[0].creature
        self.next(self.end)

    @step
    def end(self):
        print(f'The creature is {self.creature}')
        print(f'The final count is {self.count}')


if __name__ == '__main__':
    CounterBranchFlow()