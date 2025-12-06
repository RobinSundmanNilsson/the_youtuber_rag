# SQL analytics course with DuckDB - pandas and duckdb

[00:00:00] Hello and welcome to this video where we'll go into Pandas and working with Pandas together with ddb. In previous video, we have already connected Python to ddb to show you how to connect to different CSV files. But now we'll actually work with

the Quila database, which you have worked with before in.

Some lectures. This we will be able to make some exploratory data analysis using both Pandas and DAC DB inside of Pandas. That would be really cool. And then we'll use pandas plot method, which is a wrapper ~~of ~~for mat plot lib and do some simple plotting for. Quila database.

This will be ~~found ~~the foundation for later on working with evidence to make a BI report or a dashboard. That will be really fun. Let's move on to, first I'll show you where to find the Quila database inside of Kaggle. That is an Esq light file ~~fast. ~~And then afterwards I'll show you how to [00:01:00] get it inside of, ddb. But as I've shown you before how to do that now, I'll show you how to do that using Python and Pandas, that will be fun. Let's move on to the browser.

**Kokchun Giang-2:** here I am in the browser and we go into SQLite Quila sample database that you search on in Kaggle, and you can find this database. And here is some data about it. It's about DVD rental store. And it's created for my sq l in from the beginning. But this one is a sq light.

What we do is that we take this SQ light and read it into, python and read it into pandas ~~using read it. Yeah, read it into d DB ~~using Python. That will be interesting. Just download this dataset click download and then copy over the quila esq light dash quila db into your V vs code.

And we'll start over there.

**Kokchun Giang-3:** , now I'm in Visual Studio Code, and you, what you can see is this SQ light QUILA db. I cannot [00:02:00] open it. And that's the idea. ~~Now. ~~Now we will ingest this one into a duct DB database. ~~But ~~by doing that~~ we will need to, ~~I will take out my terminal ~~and I will do make the SL ~~and I will actually use the load script that I have used before.

~~Cd or I can do touch ~~touch SL load Ula ~~s ~~scale. And then here I will go in and write my script, which will basically call the attach the s scale light attach. Let's do install. I have another way ~~to ~~to ingest the data in ~~much more ~~much better way later on ~~in ~~when we'll use DLT.

I will have that in another video. Make sure to take a look into that one. But this one will suffice ~~for ~~for just doing some analysis. This is a quite simple approach that will work. Load es light and then we'll have call escalate attach, and then I would [00:03:00] need to make sure that I find the path, ~~I will be.~~

For my Jupiter notebook, I will be outside. Then I will pick out data slash es dash ula db. . And then see my colon. This is all, and now let's create that we can work with Python, UV in it. Make sure you have UV installed. UV init, and now we get a lot of different things. I will start with removing main of pie since I won't need that one.

And now I will do UV add. ~~And I want to add I pi kernel. ~~I want to add duck dd and I want to add pandas. What else do I want? I want laptop lib. I car notes for working with Jupiter Notebook. That could be in Jupi in Python, and then pandas and lib. And probably, maybe I need something else, but then ~~I would, ~~I will install it on the way.

. Clear that one. That was finished very fast. Now let us do touch I will call this duck db [00:04:00] pandas. I pin B. . Let's open this up now. What we'll do is working with Pandas and Duck db. Change my Jupiter, my kernel, and I Now I can remove the terminal Yes. And change this one to Python.

Perfect. Now let's start. ~~Import. ~~Import D db and then I will do duck DB path equals to data slash ula do duck db. , This is my duck DB path I go into. I want to go into data, I want to put in the tdb inside there.



**Kokchun Giang-4:** we will use the width statement, context manager, that when we are opening a duct DB connection, it will automatically close it down for us and [00:05:00] we'll use it to both open up a connection to Duct DB and also a, we will use it to open up an SQL script, ? What we should do here is with.

D db, no DDB dot connect. This one will open up the D DB connection will connect to data and we'll actually we have it already here. The D db path, we connected to D DB path as con. . And then comma open. SQL slash Actually I can do, yeah, SQL slash load. Ula sql. . S ingest script colon. And now I will do con [00:06:00] dot.

SQL and then I will do, this is the connection. I have the, here I use, duct db connect to duct DB path as con, and then I open up the script as ingest script. Using the connection object, I can do con do SQL, and then I will put in my ingest script, do read basically.

That means everything in my load. Ula, I will read it inside of this. This connection. . And then what I can do is, for example, when I have done that, I can do films equals to con sql from film, see my colon df, for example, and then we can take a look into films. And you can see first we create this TB and we create this TB [00:07:00] wall ahead log because yeah it it disappeared when after it has been closed to connection.

But ~~now I have this films ~~now you can see this is my films. Actually, I want to do dot head here to not show everything. And you can see there's an error here. The error is because we have already created this path. Catalog error failed to create view, actor view with name actor already exists.

That is the issue. ~~? To do that, ~~to fix this, I will ~~do, I'll ~~make this item potent by doing from import. Path ~~and I will actually do like this. I will do path and ~~I will do DB path here, DB path, and I will do dot unlink. And missing ~~is , ~~is true. What this does is that it will delete the path if it already exists.

? That means that I can run this several times and I will get the same results here. Film that had [00:08:00] three, for example, I show three of them. In this way now we have connected to ~~ddb. ~~That is really cool. And another thing I will do here is that I will call, for example, description ~~ecos ~~to con SKL.

~~Desk, ~~for example, ~~see my colon df ~~I can get the data frame from it. ~~And note ~~another thing is that ~~my ~~con doesn't work here. ~~It's ~~we have the con ~~is ~~the connection object, but you need to work with it inside of the~~ inside of the duct ~~db. ~~In the W ~~statement. What we want to do is to take out ~~desk ~~description, 'cause I picked that out.

And you can see here are all the, this is database secular. The schema is main and the name is here. Here are the, all the table names and you can see column names, et cetera. That is quite cool. And what I want to do now [00:09:00] is that I want to read in all this data into a dictionary of pandas data frames.

That means what I want, I will do description, do head you can see this pandas data frame of the ~~dis ~~description. And I want to do like this. I want to ~~read. ~~Read all data into dictionary of and as data frames. ? Because if I have a dictionary of data frames, it is a nice structure to work with I can have the data frames.

The con is that it's in memory. I materialize all of them, but the thing is it's giving me quite nice structure to work with. ~~All the, ~~with DDB dot connect and here I want to connect to data slash I can actually just pick out, the DB path, ? Because it's the same path as con.

**Kokchun Giang-5:** here we have~~ I will create ~~I [00:10:00] created this connection here. And what I want to do is actually do like this dfs equals an anti dictionary. And what I want to do is that you can see in my description here that we have the names, ? What we can do is for name in description of name.

And then what we can do here is that we can do dfs of name ~~equals to ~~equals to con S scale F string from

name, basically see my colon df. And now if you take a look into dfs. This is a dictionary of a lot of data frames. We can do DFS keys, for example. And you can see ~~here is the, a lot of the ~~here are the dictionary keys. If I want to, for example, get out the film, I can do ~~df ~~DFS of film. And you can see this is the [00:11:00] data frame for films.

We can do like dot head to just see the top of it. And if I want to have something else, for example, I want to have film actor, I can do that. This is the film actor data frame. Cool. ~~We can we can, for example ~~we can take a look into. I want to do like this, check some film related data frames.

And here you can see DFS Film Actor. ~~You can see DFS off. ~~Here we can take a look into film and we can do. Dot head of two. I don't want to show all of them. This one is quite huge. We can do dfs of film category, for example, and then take dot head here as well. And ~~we, I had film. ~~We had film actor, you can see actor ID and film id.

Here you can see film and ~~you can see, ~~actually I want to show you something else I want to show [00:12:00] you. ~~Find ula. ~~If you take a look into, ~~there's actually ~~in Kaggle when you download it, you also get this. ERD entity relationship diagram ~~and ~~where you can see how different things are connected to each other.

This will help you when you want to get some particular analysis done. ~~This is very good. ~~For example, you can see if I want to find out the film category of the film, then I need to join this film Id. With this film ID from ~~in ~~this view ~~or this, yeah. View their views. ~~Here you have Categor.

If you want to find the category, then you need to join into the category view as well. You join this one, and this one in order to get all the data from here, ~~? This one, ~~this diagram is super good to reference when we are doing different types of analysis. ? Then we can move on here.

For example, I want to have DFS of actor. ~~I can get that too. ~~And here you see [00:13:00] the names you can see dot head. ~~And ~~one example of what I just said is that, for example, DFS actor, you can see there's actor id, ? We have actor ID and we can see in the film ~~and we can see there's there should be, yeah, there's film actor actually.~~

Yeah. Film Actor has~~ action ~~Actor ID and film id. This is a bridge table, you can connect it to actor ID and film id, you can join them together to this is a bridge between film and actor. If you want both data together, let's. Take a look into actually joining. Join film related data frames using Duct db and the issue, there's an issue here and the issue is that I cannot do some, I can, if I want to do Duct db, I will show you here.

If I do duck db SQL and I do something like. [00:14:00] From my data, let's say dfs, ? If I do DFS off say actor. ~~This one worked before, ? ~~The DFS actor, it works, but from DFS actor, it doesn't work for the reason that, ddb sql, you cannot do, you cannot send in ~~a dictionary. This is ~~a dictionary.

It doesn't understand that ~~in ~~in duct db. What we need to do is we need to register this variable. To register it, we actually, we do need to do like this, ~~we need to register all ~~we need to register all data frames in Duct tv, ~~? Or all data frames that we will use. For example.~~

I can use film names. Say I want to use these, I want to use film. ~~I want to use, ~~actually, I could register all, but I don't need to. ~~I will do film actor, film category. ~~I would pick the actor as well, and I would pick the category. These, I want to register. Then ~~for, ~~to do that, I will do like this for film name, in film names, [00:15:00] ddb, register, film name, and then, this is the name that we want to register it under ~~there ~~that we can do d db scale and we can actually select from that.

Then we need to actually ~~pick, e ~~pick out the table or the data frame to register. I will do film name here. To check this, you can do something like

**Kokchun Giang-6:** to check this, you just do this Dr. B SK desk like this and do df. , Perfect. Now you can see that we have five tables. Or yeah, these are tables and we have ~~one or view, ~~1, 2, 3, 4, 5. And you can see they are inside of our data frame. Or we have exported it as a data frame, and you can see that we have 1, 2, 3, 4, 5 [00:16:00] tables.

We have all of them are inside of database temp and inside of the main schema. That is very good. Now these are the only five that we have registered. What we can do is now. We can do duct DB scale from actor directly. From actor and you can see it works good. And we can take out df.

Cool. Why did I do this? Because now when I have done this, I can. Easily work with data frames directly. I have registered them in my DDB and I can join them together. I could have done that directly in using using a connection to to, the D DB database, but this is just another way to go with it it gives more flexibility.

If you have other data frames, which isn't already in the d DB database, then you can as easily register them. Cool. Now [00:17:00] we have d db SQ from actors. Let us do some joining now. What I want to join is I want to do like this. Multi-line string first.

I will have a, yeah, I can do from actually I can do from film and I will do select

and I will also do like this. I can get more space. , Select and I will, yeah, ~~this ~~this is quite nice film f And here what I want to pick out is I want to do a star first, and now you can see I get everything from film and I want to do some joining. Left join. My strategy is that ~~I want to pick out ~~I want to pick out some film actor category and see how that works.

I want to join [00:18:00] actually four tables five tables. I want to join my film. With film actor and with actor and then with film category and with category, that is the idea. Now I have film F and I want to left, join film actor FA. This is the alias, on f Do film ID equals to fa film id.

And then I want to continue here. Left join. Not film actor, but I want to ~~act ~~take actor a on a actor ID equals to a ~~do ~~actor id on, let's see. Film actor, fa actor id. , If you just took a look into the picture we had before. Here you can see film. [00:19:00] We want to join on the film actor on the film ID and then the film actor to actor on the actor id.

By doing this, we could now, for example, I can pick out a dot first name for example. And here you can see we get all the first name and we can pick out which. Film you have played in, ? I could do like f dot title to see. Now we get first name. And actually I can concatenate it with a last name.

Let's see. I should concatenate this one as well. And now you can see this is the first name and last name. And I have which title, like which film it has played. ~~We have also, let's see. ~~I want to pick out, I can call this like as actor for example.

Now you can see which actor have played in which [00:20:00] titles. This answer that questions and also we can continue here. We can join on some more things I can do. Left. Join let's see. What do I want? I want to film category as well. Film category on film category. I call it fc.

And then we can do FC dot category. Id equals to. Or actually we haven't have category ID yet. Film id. We have, film ID and f, film id. 'cause if you take a look into this picture, we have film category, have film id film, have film id, and then category. These one we're joining together now.

By using the film ID key. . And continue. I want to pick in the category as well. And here we have the C category C and we have film [00:21:00] category of category id equals to f do category id. . And now I've gotten an error. Let's see why. F does not have a column name, category id. Yeah, because it's actually not f here.

It should be C. , Great. And now we can pick out whatever I want. For example, I want to have the actor I want to have the title. I want to have, for example, actor. I want to have let's see. I want to have categories, c name as category. Now we can see which category it has played on, ?

And I can pick out, for example, the film description. I can do that. F dot description, and we get the description here as well. And we can pick out other things ~~like ~~that exist in the row. FDOT rating, for example.

~~The, ~~these are like ~~you, ~~you want to [00:22:00] pick out all the things which you may be able to use later on, for example. I pick out, I'll just pick out a few more things. I will pick out, actor ID as well,

or actually maybe I don't need that. I have the actor. ~~I have f ~~I can pick out description. I can pick out f release year. You just pick out whatever you want. I want to pick out F dots. Let's see. Rental duration, F rental. Let me see what else we have. We have rental rate, ~~? Rental, or ~~actually we have rating yeah, rating.

I picked out rental duration. Let's see. Description. Yeah. ~~Yeah, ~~I think this is quite nice. Now ~~I ~~we picked out a lot of things here. I can actually pick out actor ID as well, a [00:23:00] actor Id just to see ~~if ~~and I also want it to be integer, ? Cost it to, ~~in, . ~~And I can call it s actor id.

Perfect. , This is my new data frame. I can call this, for example films joined data frame we can see films joined and this is the one we have, we dot head. Actually I don't want to pick all of them. Pick two and then we can take a look into films, join dot columns, for example, and we can see these are the columns that we have picked.

, Great. Let's, ~~let us now do some more EDA, ~~now do some more EDA. And now you've seen me working a lot with Duct DB in Pandas. But now let's take a look with using ~~Pandas like ~~pandas, normal methods. We can do, for example, films joined.info. This is Pandas. You can [00:24:00] see, ~~ah, ~~we have these columns.

We have, non null counts. How many ~~they ~~are. And we have here which data type ~~in ~~inside of Pandas. . And we can do a lot of other things that Pandas offers. We can do like films joined of rating for example. And here we get all the ratings and we can do like value counts. This really nice in pandas.

We can do this. And if you wanted to do this in SQL, you needed much more code, ? Take convenience methods ~~that take those ~~that work very well in pandas and you can use that. And those that work very well in SQL. You use SQL. For example, joining a lot of tables, I think using SQL is.

Far superior ~~from than ~~using pandas and also grouped by and other things I like to use. Duct DB and SQL. We can take for example, we can continue here. We can do films joined and pick out the rental duration and do like value counts here as well. And you can see [00:25:00] how that looks like.

We can also do. Films ~~join. This is quite nice, ~~but describe and see what happens here. Actually we only get the rental duration. Yeah, that's fine. Actor ID ~~is ~~doesn't make sense ~~to ~~to describe on but rental duration because~~ what the, ~~what describe does is that it picks all the numerical columns and will make summary statistics based on that one.

What I want to do is pick out ~~the ~~the rental duration here. , This is for rental duration, and I want ~~to do t ~~to transpose it. ~~Or actually, ~~yeah, ~~this ~~this is good. I think. Yeah, you can see the mean, how much the rental duration is there. Standard deviation mean 25% percentile, et cetera.

, Great. Seven days max and three days is minimum. , Cool. Can we start to answer some relevant questions? For example, we can ask a [00:26:00] question like, which top 10 actors have played in most films, for example. And to find this, we could, for example, do like this. We can do actor ID films equals to duct db.

SQL

and we can do select actor id. Or actually we can, we don't need to do actor id. We can do actor. I will actually remove this one first and I will give an appropriate name later on. What is the, let me see the. Films joined. Films joined dot dot columns. , It's called Actor Only. Select Actor from Films [00:27:00] joined.

Note that we don't need to register this one because it's already a data frame. A data frame we can put in directly. We needed to register the the one before because they were ~~in ~~in dictionary, ? Here you get all the actors and we take dot DF, we get the data frame back.

, I want to find out number of films that you have played. . What we can do is to ~~group by, we can ~~group by the actor id, then we know we get all the unique actors ~~and we do let's see. ~~And we need to find a aggregation function as well. And what this aggregation function that we'll use~~ we can do ~~count, count, star.

Oh, sorry. Count star as number films. Let's see why count, star, number film actor must appear in group by. ~~Oh, ~~then I pick actor instead. . Like this. And what I should do is to do actually ~~yeah, ~~actor. And ~~actor ID or actor? ~~Actor should be enough. ~~Yes. ~~And I [00:28:00] want to order by~~ order by, let's see, what do I want?~~

~~And ~~number films descending. , Great. Now we have the top. And then what we do is, for example, if we want to have the 10, we can do limit 10. And now we have 10. Or instead of limit, we could take this out as a data frame and we can do something like actor films equals to this one. And then we can have actor films head of 10 and we get the same results here.

Great. And what can you do with this? You can as easily do some visualizations if you want. Visualize this. . Then we have actor films. We can do dot head of 10. We take out the top 10 and then we can do dot plot. And this one will use mat plot lib in the background. Kind is bar H.

Actually I want to have~~ h ~~horizontal bar. And what we'll do is [00:29:00] x equals to actor y equals to number films. . And now we should get a. Horizontal bar chart. Perfect. And I can call it X and I want to do X dots invert y axis. , Perfect. Now it looks much better. And now we can just do title equals to.

Top 10 prolific actors, for example, and you can see this one, and we can do like X label and Y label as well. X label equals to number of films. Number of films and we can do y label. Yeah, we have the after there already, perfect. Good. Yes. That was quite cool. And you can [00:30:00] continue and find something else.

For example, we can visualize number of movies ~~in ~~in each ~~gen ~~genre. With this, you can see, ~~we can use, ~~we can visualize and there's a lot of other things that you can visualize, but I won't show it to you. Instead~~ you can ~~you can explore yourself and feel free and explore a lot in the database because there's a lot of things to pick out.

There's a lot of, you can take a look into different revenues. You can just find a question that you want to answer, and then try to use SQL and pandas to answer it. After this, make sure that you commit and then push your changes to GitHub. 

in this video you have gone through using DDB in Pandas and we have used both~~ something some ~~methods in Pandas and we have used DDB ~~in ~~in Pandas as well, that we're able to. Do analysis in a very convenient way and in a way that will help us a lot. And [00:31:00] then we did some very simple visualization to show you that after you have the result in the pandas data frame~~ you can, ~~you have basically.

All the tools that are available in your normal Python ecosystem. That is super cool and I hope that you've learned a lot and see you in the next video where we'll go into DLT data load tool to actually. Load the data directly from the Quila database into Duct db that you actually have it there instead of doing what we the simple approach that we chose, that used the attach Esq Light, the attach.

Yes. See you in the next video. Thank you. Bye.

