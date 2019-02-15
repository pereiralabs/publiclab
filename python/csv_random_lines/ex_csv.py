#This script reads a CSV file, then chops some columns and prints some random lines

#Setting the locale
# -*- coding: utf-8 -*

#Libraries section
import csv
import random

#Vars section

#List of files you want to extract random rows from
file_list = ["/home/foo/file-01.csv", "/home/foo/file-02.csv" ]

#List of columns you want to extract from each row
included_cols = [1, 2, 3, 5, 7]

#Number of rows you want to randomly extract from each file
desired_num_results = 600

#The min number of rows between all files
#e.g.: if you have 3 files (fileA has 30 rows, fileB has 40 rows and fileC has 50 rows) then your total_entries will be 30
total_entries = 840


for file in file_list:

    #Creating a list of random line numbers
    #It starts at 2 to skip the header at line 1
    rows_list = [random.randint(2,total_entries)]
    i = 2
    while i <= desired_num_results:
        #Appends random elements
        rows_list.append(random.randint(2,total_entries))
        i = i + 1
    #Prints the sorted numbers for debugging
    #print rows_list

    with open(file) as csvfile:
        cdrreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        j = 1
        #Prints only the selected columns of the randomly choosed rows
        for row in cdrreader:
            if j in rows_list:
                content = list(row[x] for x in included_cols)
                print '"' , '","'.join(content) , '"'
            j = j + 1
