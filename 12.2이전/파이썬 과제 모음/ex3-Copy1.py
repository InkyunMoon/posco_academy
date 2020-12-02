import sys

print(sys.argv)

fr = open(sys.argv[1], 'r')
fw = open(sys.argv[2], 'w')

for line in fr:
    fw.write(line)
    
fr.close()
fw.close()