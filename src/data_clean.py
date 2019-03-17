# data_clean.py
# Cleans the provided dataset of region_id
def clean(inFilename,outFilename):
    with open(inFilename) as infile, open(outFilename,mode='w') as outfile:
        line = infile.readline()
        while line:
            toList = line.split(',')
            cleaned = toList[0] + ','
            toList = toList[2:]
            for i in range(len(toList)-1):
                cleaned = cleaned + toList[i] + ','
            cleaned = cleaned + toList[len(toList)-1]
            outfile.writelines(cleaned)
            line = infile.readline()

if __name__ == '__main__':
    clean('dota2Test.csv','out.csv')