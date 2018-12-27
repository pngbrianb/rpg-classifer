# build a single function to export and run in a single cell! So convenient!
def rpg_classify(suburl):
    print("Importing Modules...")
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    from sklearn.naive_bayes import MultinomialNB

    import requests
    import time
    import json
    from IPython.display import clear_output
    
    # first let's build our model. It will read in our data:
    clear_output()
    # first let's build our model. It will read in our data:
    print("Reading Data...")
    dfpp = pd.read_csv('./data/pen-and-paper.csv')
    dfvg = pd.read_csv('./data/video-game.csv')
    
    dfpp['sub'] = 1
    dfvg['sub'] = 0
    df = pd.concat([dfpp,dfvg])
    df.reset_index(drop=True,inplace=True)
    df.drop(columns=['name','url'],inplace=True)
    
    def text_blob(row):
        if type(df['comments'][row]) == str:
            return df['text'][row] + df['comments'][row]
        else:
            return df['text'][row]
    
    df['alltext'] = [text_blob(i) for i in range(len(df))]

    X = df['alltext']
    y = df['sub']
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)
    
    # we'll vectorize the data too, but that won't get a print statement
    cv = CountVectorizer(max_features = 4000, stop_words = 'english')
    X_train_cv = pd.DataFrame(cv.fit_transform(X_train).todense(),
                          columns = cv.get_feature_names())
    
    # second, let's instantiate and fit our Naive-Bayes model
    clear_output()
    print("Building Model...")
    nb = MultinomialNB(alpha=0.01)
    model = nb.fit(X_train_cv, y_train)
    
    # And now we'll repeat our earlier steps to read in our posts
    clear_output()
    print("Reading Posts...")
    posts = []
    after = None
    headers = {'User-agent':'BBLab03'}

    # iterating any more than this is a waste.
    for i in range(40):
        # our API scrape uses Reddit's 'after' parameter
        if after == None:
            params = {}
        else:
            params = {'after':after}

        url = suburl+'.json'
        res = requests.get(url, params = params, headers=headers)
        # A quick check that everything is coming through alright
        if res.status_code == 200:
            the_json = res.json()
            # add to posts
            posts.extend(the_json['data']['children'])
            after = the_json['data']['after']
        # Print any error codes that come up
        else:
            print(res.status_code)
            break
        # Print something to make sure progress happens while the program runs 
        if i % 10 == 0:
            headers['User-agent'] = 'BBLab03-'+str(i)
            clear_output()
            print("Reading Posts...")
            print(str((i)*25)+" posts so far.")
        # make sure not to overload Reddit with requests!
        time.sleep(.25)
    
    clear_output()
    print("Formatting Posts...")
    # that read in all our posts. Now we extract what info we need into a dataframe
    list_of_lists = []
    # iterate over the posts
    for i in range(len(posts)):
        # fill in our desired fields
        sub = posts[i]['data']['subreddit']
        name = posts[i]['data']['name']
        title = posts[i]['data']['title']
        body = posts[i]['data']['selftext']
        suffix = posts[i]['data']['permalink']
        url = 'https://www.reddit.com'+ str(suffix)
        text = title + body
        row = [sub,name,text,url,None]
        # Here's where I catch duplicates
        if row not in list_of_lists:
            list_of_lists.append(row)
    # Here I put my list into an easy-to-use DataFrame!    
    df = pd.DataFrame(data=list_of_lists,columns=['sub','name','text','url','comments'])
    
    # iterate over the dataframe
    for row in range(len(df)):
        # finish formatting the url address
        url = str(df['url'][row]+'.json')
        res = requests.get(url,headers=headers)
        the_json = res.json()
        # empty list for depositing our comments
        comment_list = []
        # A quick check to skip over comment-less posts
        if the_json[1]['data']['children']:
            for comment in range(len(the_json[1]['data']['children'])):
                # the Reddit API doesn't give body text for more than 50 comments on a single
                # post
                if comment <= 50:
                    try:
                        comment_list.append(the_json[1]['data']['children'][comment]['data']['body'])
                    except KeyError:
                        print('We got some invalid comments!')
                        print("row: ",row,'; comment: ',comment)
                        break
                    df['comments'][row] = comment_list
        # print to ensure the program runs, delay it enough that it doesn't clog up the series
        # of tubes
        if row % 10 == 0:
            clear_output()
            print("Reading Comments...")
            print(str(row)+" rows down!")
        time.sleep(.03)
    
    clear_output()
    print("Formatting Text...")
    def text_blob(row):
        if type(df['comments'][row]) == str:
            return df['text'][row] + df['comments'][row]
        else:
            return df['text'][row]

    df['alltext'] = [text_blob(i) for i in range(len(df))]
    
    clear_output()
    print("Classifying Subreddit...")
    X = df['alltext']
    X_cv = pd.DataFrame(cv.transform(X).todense(),
                         columns = cv.get_feature_names())
    
    # since 1=pen and paper and 0=video game, rounding the mean of all posts' predictions
    # will give the model its classification!
    subscore = model.predict(X_cv).mean()
    if subscore >= 0.5:
        print("This is a pen-and-paper RPG subreddit!")
    elif (subscore >= 0) & (subscore < 0.5):
        print("This is a video game RPG subreddit!")
    else:
        print("Something has gone horribly wrong!")
        