def word_search(doc_list):
    # list to hold the indices of matching documents
    # Iterate through the indices (i) and elements (doc) of documents
    tokens = doc_list.split()
    print(tokens)
    # Make a transformed list where we 'normalize' each word to facilitate matching.
    # Periods and commas are removed from the end of each word, and it's set to all lowercase.
    normalized = [token.rstrip('.,').lower() for token in tokens]
    # Is there a match? If so, update the list of matching indices.
    print(normalized)


l = ['The Learn Python Challenge Casino', 'LOLOL', 'Really.']
word_search(l)
