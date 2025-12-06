# SQL analytics course with DuckDB - dlt to load sakila data from SQLite into DuckDB

[00:00:00] Hello and welcome to this video where welcome into DLT Data Load Tool. This video is for. Using DLT to load the data in Quila database the SQ light database into a duct DB database. And in the previous video, we actually ~~used we ~~used load and attach using the ESQ light extension in duct DB to do that.

However, ~~using d ~~it's good to learn about DLT. Since DLT, you choose the source and the destination and they can be different. For example, now we just picked from Esq Light into Duct db, but it could be from cul light into MyQ. It could be cul light into. Postgres or other sources and destinations.

~~That ~~the thing is that it standardizes how to load the data and this is really useful to learn. And if you want to learn more about ~~DDLT, I have used DLT to also ~~I have video where I use DLT to [00:01:00] load data from API into Snowflake, into D db. That is also really interesting that the DLT is really good at solving the el part extract load of the ELT, ?

This is really useful to learn and to know about, but I will just go into the very basics here for cul light into duct db. .

**Kokchun Giang-1:** , Let's start in Kaggle. Here I searched for Esq Light Quila sample database. This is the one that we will use and you can just download it. Click download here and take out, ~~esq DB ~~esq dash quila db file. Basically this is a sample database for fixtures database designed to represent the DVD rental store.

And this is original developed for my SQL. But now we will use it and ingest it from ~~cul light into ~~into, the source is ~~Escal light ~~and the. ~~And ~~destination is Duck db. I'll show you how to do [00:02:00] that. ~~Click download here. ~~Make sure that you are logged into Kaggle using, for example, your Google account.

**Kokchun Giang-2:** , here I am in VS code and I have my ESQ Light dash QUILA db. Yes. I cannot open it and this is the only thing that I have. Let's start from scratch UV in it that we have our. Project started our main do pi, I remove this one. I don't need the main dot pi. What I will need is to do touch I will call this load re Esq db do pi.

, I will go from sla I'll go from Escal light to D db. This is the file that I will use and I will use uv. I need to install some things UV add first, IPI kernel, because I will actually use some Jupyter Notebook later on to. Test it out. But first we'll [00:03:00] use DLT. Here I will use DLT and I will need to have install it in this way because I will need something called SQL score database.

This is needed for doing the connection to SQ light. That's the thing. And then actually, I think this might be everything. I only installed this one. Oh, there is some issues here. Or not requirement named DLT matches project name DLT, ? That was a mistake of me to call this DLT. I will post the video and I will change the name of my project name.

**Kokchun Giang-3:** , basically what I did was to go into PI Project Omo and change this to DLT Quila instead, ~~that it, ~~that I can install this one. I could do uv ADD IPI kernel D-L-T-S-K-L database without problems now. It works. Let's now go into this script here and let's type it out.

It is very simple to [00:04:00] work with this. You have import DLT, and then you have from~~ actually I should change here to ~~my Python. ~~To my ~~virtual environment ~~and ~~here from DLT sources do SQL database, import SQL database. Remember that I need this one to work with SQ light. ~~I needed to or ~~I needed to connect to ES light more you can use it to connect to other ones as well.

~~Here, let us you ~~here you can~~ I will ~~also take in path lip import path and the path lip. I will use it ~~to ~~to create the paths to my escal light and also to my duct DB path. , I'll do Escal light path equals to path of ~~done the file. Then ~~the file, you'll ~~get this exactly the, I'll ~~get the path to this file, and then I'll do dot parent, and now I will be in the DLT.

~~And ~~then I will do slash data. That means I will go into the data path, DLT data path, [00:05:00] and then I will go into slash Esq light dash Quila db. ~~And ~~now I have the path to the quila. I have this escal light path, and what I could do is that I actually would call this data path and I will remove this part.

Why do I do that? Because then I can reuse data path and I will call this now escal light path equals to data path and then paste this one in. Now I go to. Escal dash QUILA db, and then I can copy this one shift AL down button. And now I will have, instead of Escal light path, I will have duck DB path.

. And it is the same data path, but here I will just name it differently. What do I want to name it? I will call it sa. , These are. You can see this as this is the source S Scale light [00:06:00] path. This is the source path and this is the destination path. ? Based on this one, we'll actually create our source, SGL L database.

And here we'll do credentials equals to basically like this F string. And I'll have Escal light. You don't have any credentials in escal light. 1, 2, 3. And here we put in the escal light. Path. That is it. And then we do let's see comma, we choose which schema we pick from. Schema equals to main.

I'll go from the main schema and because there all my views are there in the main schema. And I want to create a pipeline. This is how DLT works. They always have this pipeline for running the data from running from. ~~A ~~a source into a destination. What I do is pipeline. Pipeline object DLT do pipeline.

And here I [00:07:00] will have a pipeline name, and the pipeline name is called Ula Es escal Light. I'll call it Ula. This pipeline name. And then I will have destination equals to here. I'll use. Let me just take some more space here. DLT destinations Duck db, and here I will have string of my duck DB puff.

Datasets name. This is the schema, basically. I will call the staging because I want my data to land in the stagings schema. This is the staging layer, and if I want to transform the data, I will have another layer called the refined layer. Let us start with this. This is our pipeline. And then what we do is load info equals to [00:08:00] pipeline run.

And I will pick the source because this is the you have the pipeline which defines where the data will land, and we put in the source for it. The data that will be sent. , Now I will also do disposition. The simplest one is to replace it, which means that yes, ~~we will ~~we will get the data from the source, put it into our destination, and if there's data there before we just replace it.

? And then we'll do print load info. And my friends, this is it. What we should do now is to go in and run this one. I will go into my terminal and do UV run Python load, ula, escal, light Tech, pb.

Let's see.

**Kokchun Giang-4:** , that error before just said ~~we need to install ~~we need to install DLT duck db. [00:09:00] Uv UV add DLT duck DB like this. . After we have done that, yeah, it just installed duct DB basically. Now we can run it. We can do uv run Python load ULA escal T db.

I run this one and you can see something happened. We have T db dot well file disappeared. , Then let's connect and see. Let's create another a Jupiter notebook here now. I'll call this, test ULA dot IPD. Now it's just to check our ~~ula. Check ~~ULA. Change my, my virtual environment and now change this one to Python import Duck db.

And. What we need to do is very simple. We just need to connect. With ddb, [00:10:00] do connect where is the file? Where? It's in data. Data slash ULA do ddb Perfect. Colon on ask con for connection. And if you've seen my previous video I did a connection with and, and that we did a lot of exploratory data analysis.

I won't do that. I'll just check that it works. This, I'll do description equals to con SGL desk. . See my column TF to export the data frame. Now I can just type description. Module not found mpi. , Then let's install it. UV ADD mpi. Actually UV add pandas. Then it will have a MPI as well.

, MPI and Pandas you saw. , Now you can see something happens.

Yes. Perfect. Now you can see that our. [00:11:00] Database is populated with a lot of tables we have here as the database name, and then we have schema that is the staging. We chose to have it in the staging schema. That was the dataset name. Remember here we have the dataset name as staging, and then we have some meta tables which will keep track of the versions, the pipeline state, and the number of loads that we have done.

~~These ~~these are the ones corresponding to DLT ones. You have underscore here in front, and the rest are the quila tables. You can see here, actor address category, city, et cetera, code. ~~If you want to, for example, we can do dot head here. ~~I don't need to show everything, but we could, for example in the last video I showed you how to.

Store the data into dfs, ? And then register them into tdb I can show you how to do that. Similarly here now but what I will do is [00:12:00] I'll start with to storing data into dictionary of the Fs, basically. To do that. We will do here, DFS equals to an ~~empty ~~empty dictionary.

And then I'll do with duck DB dot connect here. I will do data slash ULA duck DB as con, and after that ~~I will do ~~I will do something in style with this I will do four.

Basically we have the desk, ? Desk we have name and we can do four name in description of, what is it called? It's called ~~name ~~name. And then we can do something with this. D Fs of name equals to con sk. And here I'll do sring and I'll do from staging dots name, see my colon. [00:13:00] DF. This one.

Now I will have a lot of things in my Ds. But you can see we have these underscores as well, but I don't want to have those. I will say for name in. What I can do is let's see. Just simply like if. Underscore if for example, like this name of zero is the first character if this is not equals to underscore, then we'll do this.

Otherwise we won't do this. , Now you can see they have. Or actually we have DLT load id. Yeah, this is part of the actor's table, that is fine because that is how DLT works to keep track of all the things. But that is fine. I'll see, for example, here, if we pick out actor, you can see here we have this.

DLT load ID as well, and a DLT id. [00:14:00] This is just for, to keep track of things that you can do, for example, incremental loading and other things. Here, just take a look into that one and then we will register all.

Actually we can register it all directly in this for loop, actually for statement. What we can do here is to do ddb dot register

and register. Let's say the name, DFS of name, as simple as that. , To check, ~~we can do ~~we can do duck db, do sgl, and we can take a look into, if we do desk, you can see, ~~we can do here. ~~And you can see now we have registered all of these into our. ~~D DB into ~~D db. You can see they are inside of the temp [00:15:00] database, temporary database, and the main schema.

And we have all of them here. That means we can do something in style as this one. D db spl, we can do from actor for example, and then vf. Perfect. And here you can say you can pick out different things. From actor, for example, ~~select ~~select first name for example. And here you get all the first name.

Kamal, last name. Cool. Now you see it works. We have loaded the data from escal Light into Duct db. And then afterwards we have used this test ULA to actually check out all the tables. And ~~we saw that and we we did some ~~we store them in DF that we can have this nice data structure to see the data frames.

And we have also done registering. Here we register them to DDB that when you have it in~~ into ~~[00:16:00] ddb ~~and ~~you can very easily work with it and you can do different joints, et cetera with several tables. Without using the con object, we don't need to connect to the duck DB all the time.

We don't need to write this type of syntax all the time with duck db do connect, ? That is really cool. Let's move on and you should~~ make sure that you ~~make sure to commit and push your changes to ~~get the ~~GitHub. Yes, 

yes. In this video we have worked with DLT to load the data from sq light quila into ddb. And then when we have it inside of ddb we use the Jupyter Notebook to test it out and to do some registering and, and then afterwards showed that it worked with~~ using, ~~doing some queries on some tables ~~that ~~that was super nice.

And I hope that you learned a lot in this video and that you start to enjoy DLT and see the power of it to standardize the way of moving data from one place to another one. It's really nice for [00:17:00] doing some different types of migrations. And it's super cool. Thank you for watching this video and see you in the next one.

Bye.

