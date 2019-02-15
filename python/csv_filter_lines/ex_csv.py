#This script reads a CSV file, then return lines based on a filter

#Setting the locale
# -*- coding: utf-8 -*

#Libraries section
import csv

#List of files you want to extract rows from
file_list = ["/home/foo/Downloads/20180514.csv"]

#List of columns you want to extract from each row
included_cols = [5,2,1]

#Now let's iterate through each file on the list
for file in file_list:
    #For each file:
    with open(file) as csvfile:
        cdrreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        j = 1
        #For each row in a file:
        for row in cdrreader:            
            #Checks if condition is satisfied
            if row[1] == "alô":                
                #If it is, then the selected columns are retrieved from the row
                content = list(row[x] for x in included_cols)
                print(content)                
            j = j + 1
