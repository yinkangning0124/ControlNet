from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg'] # target, image
txt = item['txt']
hint = item['hint'] # source, control signal
print(txt)
print(jpg.shape)
print(hint.shape)
