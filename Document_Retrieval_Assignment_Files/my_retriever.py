import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting

        # Method performing retrieval for specified query

        #Getting max document id which should be 3204
        docids = []
        for word, val in index.items():
            #store all the document ids in an array
            docids.append(max((list(val))))

        #maximum document id
        self.max_doc_id = max(docids)
        
        #Calculating TF
        #Create nested dictionary-> keys:doc_id & values:(word:count)
        doc_dic_tf = dict()
        #create empty nested dictioary with key of documet ids
        for doc_id in range(self.max_doc_id):
            doc_dic_tf[doc_id+1] = dict()

        #Dictionary -> docid:keys & word:count
        for word, val in index.items():
            for docid, count in val.items():
                doc_dic_tf[docid][word] = count

        self.documend_id_dic = doc_dic_tf

        #for each document get the length of the document
        binarySizeVector = dict()
        tfSizeVector = dict()      

        for docid, val in self.documend_id_dic.items():
            binaryArray = []
            tfArray = []
            for word, count in val.items():
                binaryArray.append(1)
                #put in the array count*count to add them up and get squared
                tfArray.append(count*count)
                binarySizeVector[docid] = math.sqrt(sum(binaryArray))
                tfSizeVector[docid] = math.sqrt(sum(tfArray))

        self.doc_size_binary = binarySizeVector
        self.doc_size_tf = tfSizeVector

        #document frequence to calculate idf
        #number of documents containing the word
        documentFrequence = dict()
        #By calculating docid, df can be calculated
        for word, val in index.items():
            #make the val as string
            val = str(val)
            #number of val is same as df
            tokenize = val.split(",")
            #create a new dictionary - word as key and df as value
            documentFrequence[word] = len(tokenize)

        #idf = documentsize/df(from documentFrequency)
        idf = dict()
        for word, df in documentFrequence.items():
            #key -> word, value -> idf
            idf[word] = math.log(
                self.max_doc_id/df)

        self.idf_vector = idf

        #calculate tfidf for document vector
        tfidf_dic = dict()
        #create empty nested dictionary-> key:docid
        for id in range(self.max_doc_id):
            tfidf_dic[id+1] = dict()

        #create dictionary - key:word & value: (word:tfidf)
        for docid, val in self.documend_id_dic.items():
            for word, tf in val.items():
                tfidf_dic[docid][word] = tf*self.idf_vector[word]

        self.tfidf_vector = tfidf_dic

        #calculating the length of tfidf document vetor size
        tfidf_length_dic = dict()
        for docid, val in tfidf_dic.items():
            #the array should be empty for next document
            tfidf_arr = []
            for word, tfidf in val.items():
                #add the value to the array to add them all together later
                tfidf_arr.append(tfidf*tfidf)

            tfidf_length_dic[docid] = math.sqrt(sum(tfidf_arr))
        
        self.tfidf_length = tfidf_length_dic
   
    #BINARY METHOD

    #To get the size of queyr vertor
    def length_query_binary(self, index):
        binaryArray = []
        for key, val in index.items():
            #because value = 1, (1*1) == 1
            binaryArray.append(1)

        #square of sum of the word's value
        totalSum = math.sqrt(sum(binaryArray))

        return totalSum

    #get cosine similarity of a query with all 3204 documents
    def cosForbinary(self, query):
        ranking = dict()
        for docid in range(self.max_doc_id):
            #docid start from 0 so use docid+1
            #Get the document that we wanna compare with a query
            comparedDoc = self.documend_id_dic[docid+1]
            # array should be reset for the next document
            vector = []
            for word, val in query.items():
                # The word in a query should exist in the dcoument
                # doc_count*query_count are both 1 so 1*1 = 1
                if word in comparedDoc.keys():
                    vector.append(1)

            #calculating coine similarity
            similarity = (sum(vector) / (
                (self.doc_size_binary[docid+1])*(self.length_query_binary(query))))
            #create newdictionary-> keys:docid & value:similarit with a query
            ranking[docid+1] = similarity
            
        return ranking
    
    #Get the top ten dcouments for a given query 
    def bestTen(self, query):
        #empty dictionary for to store docid and similarity to sort in order
        newDic =  []
        #Get a dictonary: keys:docid & value:similarity with a query
        cosDic = self.cosForbinary(query)
        for docid, similarity in cosDic.items():
            #swap the postion to sort similarity in descending order
            newDic.append((similarity, docid))

        #sort in descending order with it's similarity    
        top_10 = (sorted(newDic, reverse=True))[:10]
        #And get only document id of them
        finalresult=[x[1] for x in top_10]

        return finalresult

    # TF METHOD --

    #Caculating the length of query vector with tf weighting
    def length_query_tf(self, query):
        vectorArray = []
        #put the count*count in to get the length of vector
        for word, count in query.items():
            vectorArray.append(count*count)

        #squre of all the valeus added up
        totalSum = math.sqrt(sum(vectorArray))

        return totalSum

    #similarities for each document using tf method
    def cosForTf(self, query):
        similarity = dict()
        #similiarity for each document wiht a given query
        for docid in range(self.max_doc_id):
            comparedDoc = self.documend_id_dic[docid+1]
            #should be reset for the next document
            vector = []
            for word, count in query.items():
                #query word should be in the documentvector
                if (word in comparedDoc.keys()):
                    vector.append(
                        (comparedDoc[word])*count)

            result = sum(vector) / \
                ((self.doc_size_tf[docid+1])*(self.length_query_tf(query)))
            #similarity value witht he given document id
            similarity[docid+1] = result

        return similarity

    #Get the top ten dcouments for a query
    def bestTenForTF(self, query):
        #empty dictionary for to store docid and similarity to sort in order
        newDic = []
        #Get a dictonary: keys-> docid & value -> similarity with a query by using tf method
        cosDic = self.cosForTf(query)
        for docid, similarity in cosDic.items():
            #swap the postion to sort similarity in descending order
            newDic.append((similarity, docid))
            
        #sort in descending order with it's similarity
        top_10 = (sorted(newDic, reverse=True))[:10]
        #And get only document id of them
        finalresult = [x[1] for x in top_10]

        return finalresult

    #TFIDF METHOD --

    #Query vector with tfidf applied
    def tfidf_query(self, query):
        query_length_vector = dict()
        #querycount is tf for query word
        for queryword, querycount in query.items():
            #queryword should be in the document
            if queryword in self.idf_vector.keys():
                query_length_vector[queryword] = querycount*self.idf_vector[queryword]

        return query_length_vector

    #calculating the length of query vector for tfidf method
    def tfidf_query_size(self, query):
        idf_query = self.tfidf_query(query)
        query_size = []
        for word, tfidf in idf_query.items():
            query_size.append(tfidf*tfidf)

        #add the value to the array to add them all together later
        length_of_query_vector = math.sqrt(sum(query_size))

        return length_of_query_vector

    #calculating cos similarity with tfidf method
    def cos_tfidf(self, query):
        tfidf_dic = dict()
        for docid in range(self.max_doc_id):
            #clear the array for next document
            caculation = []
            for queryword, querytfidf in self.tfidf_query(query).items():
                #Query wrod should be in the document
                if queryword in self.tfidf_vector[docid+1].keys():
                    #use array to sum the values altogether later
                    caculation.append(querytfidf*(self.tfidf_vector[docid+1][queryword]))

            #cosine smilarity calculation
            similarity = ((sum(caculation)) /
                          ((self.tfidf_length[docid+1])*(self.tfidf_query_size(query))))
            #create dictionary: key:documentid & value:similarity
            tfidf_dic[docid+1] = similarity
        
        return tfidf_dic
        
    #Get the top ten dcouments for a query
    def bestTenForidf(self, query):
        #empty dictionary for to store docid and similarity to sort in order
        newDic = []
        #Get a dictonary: keys-> docid & value -> similarity with a query by using tfidf method
        cosDic = self.cos_tfidf(query)
        for docid, similarity in cosDic.items():
            #swap the postion to sort similarity in descending order
            newDic.append((similarity, docid))

        #sort in descending order with it's similarity
        top_10 = (sorted(newDic, reverse=True))[:10]
        #And get only document id of them
        finalresult = [x[1] for x in top_10]

        return finalresult

    def forQuery(self, query):
        #Different methods will be used with diferent term weighting
        if (self.termWeighting == 'tf'):
            top_10 = self.bestTenForTF(query)
        if (self.termWeighting == 'tfidf'):
            top_10 = self.bestTenForidf(query)
        if (self.termWeighting == 'binary'):
            top_10 = self.bestTen(query)

        #print(top_10)
        return top_10



