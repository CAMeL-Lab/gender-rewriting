def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def clean_and_write(data, path):
    clean_data = [line for line in data if line]
    with open(path, mode='w') as f:
        for line in clean_data:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    src_data = read_data('train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B')
    tgt_data = read_data('train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B')

    clean_and_write(src_data, 'train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean')
    clean_and_write(tgt_data, 'train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean')

    src_data = read_data('dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B')
    tgt_data = read_data('dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B')

    clean_and_write(src_data, 'dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B.clean')
    clean_and_write(tgt_data, 'dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B.clean')

    src_data = read_data('test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B')
    tgt_data = read_data('test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B')

    clean_and_write(src_data, 'test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B.clean')
    clean_and_write(tgt_data, 'test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B.clean')
