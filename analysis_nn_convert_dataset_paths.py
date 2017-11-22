

print("reading train_list.out")
with open('train_list.out') as f:
        lines = sorted(f.read().splitlines())
        new = [line.rsplit("/",1)[0] for line in lines]

with open('train_list.out','w') as f:
        [f.write(line+'\n') for line  in new]


print("reading validation_list_real.out")
with open('validation_list_real.out') as f:
        lines = sorted(f.read().splitlines())
        new = [line.rsplit("/", 1)[0] for line in lines]

with open('validation_list_real.out', 'w') as f:
    [f.write(line + '\n') for line in new]


print("reading test_list.out")
with open('test_list.out') as f:
        lines = sorted(f.read().splitlines())
        new = [line.rsplit("/", 1)[0] for line in lines]

with open('test_list.out', 'w') as f:
    [f.write(line + '\n') for line in new]