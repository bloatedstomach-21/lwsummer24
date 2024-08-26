from googlesearch import search
searchtext=input("enter the search text")
def searchresults1(searchtext):
	print(list(search(searchtext,num_results=5)))	
searchresults1(searchtext)