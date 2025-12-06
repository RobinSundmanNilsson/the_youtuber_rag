# SQL analytics course with DuckDB - CRUD operations tutorial

[00:00:00] Hello and welcome to this video where we'll go into crowd operations hands-on. Here we will create, we will read, we'll update and we'll delete. We'll create data using the insert statement, and then we'll do read with select. And we'll do update with Update and delete data. With delete. Here in this tutorial here, I will go through using, glossary data. We'll create the glossaries and their descriptions using insert with~~ some ~~some explanation of~~ like ~~programming concepts, ~~some of ~~database and d db. ~~Then it, ~~it should feel a little bit, like you are working with some kind of glossary application, which you have in the backend and the database which ~~you in the future, ~~you can connect to an API as well.

Here for simplicity, ~~we'll, ~~or ~~for ~~because we have come ~~into ~~this part, we'll work in the database side using DDB and SQL. Let's move on directly to Visual Studio [00:01:00] Code.

**Kokchun Giang-1:** Okay, here I am in an empty visual studio code. Here you can see crowd operations, and we start with creating a duct db. Or actually we'll create some SL scripts first. Make the actually I will put all the s scale scripts in the root here, I don't need to have ~~a ~~another folder called S Scale, or we can have make the S scale.

And CD into SQ and inside of SQL folder, I will create I will do touch, what do I need to create statement, do sq I will have alter, do SQ as well and I will drop that a scale and I will have my crowd operations. Crowd. Great. Do es scale to crud? Read scale. Here three crud.

Update. Do s scale [00:02:00] for crud. Delete scale. Okay, I create all these S scale files. Here you can see if I do Ls. These are the s scale scripts that I've created. We'll work with all of this in this video. Now we'll have a. DDB database. CD go up. Cd dot means going up apparent.

Here I will do DDB dash ui and I will call this glossary and ddb. Okay, this opens up, you can see we get this glossary dot duct db. And I will put this here. I will actually decrease this a little bit and decrease this a little bit more that we can see properly see our database.

'Cause the idea is that I will create a code in the left here and then ~~I will say ~~I will copy it into this cells here and I will. Random and you will see the [00:03:00] results. Let's create a new notebook. And this notebook I will call glossary. Okay, glossary. And you can see it's connected to glossary.

Okay, starting here, we'll go into the create statement.

**Kokchun Giang-2:** Okay, let's us start with actually ~~creating the ~~creating the schemas. What I will do here is create schema. ~~I'll remove this one. ~~Create schema, if not exists, database. ~~Okay. I will copy this one. ~~I will create this one. ~~And I will actually do similar. I'll copy it. ~~Create schema, if not exists, programming.

I'll have two ~~s ~~schemas, I'll have the database schema and I will have the programming schema. I'll save this one and I will also, ~~I'll copy it first, ~~paste it here, and I'll run this one. ~~You say ~~you see that zero rows return, two statements run. ~~Okay. ~~If I run it again, you can see that it's still running because it's ~~item ~~potent.

Because I ~~have done, ~~I have put in this, if not [00:04:00] exists. It does nothing ~~when it's ~~if it already exists. If I do desk now, you can see what happens. ~~We have our main here. ~~Let's ~~see. We should actually do. ~~Check out which ~~s ~~schemas we have. From information schema. Schema, okay.

~~And now remove this one. ~~Here, these are the ~~s ~~schemas that we'll have. You can see we have glossary. ~~We have ~~we have, yeah, exactly. Inside of glossary. The other ones are the systems, we don't need to care about those. In the system we have this information schema, right by default, ~~and this main, ~~and this PG catalog ~~and this main, PG catalog.~~

~~It's the, ~~it's similar to like in Postgres. Dr. Db has taken a lot of inspiration from Postgres. ~~Here we have ~~but we only need to care about this for glossary. We have database, we have main, we have programming, right? Copy this inside here as well. This is what I use to check the schemas.

Check schemas. Schemas, remember one schema is corresponding to a, it's a logical grouping, you can see it as a folder of several [00:05:00] tables and views. Okay, now we will do one thing, is that when we are creating new glossaries, ~~we in, ~~in duct db we need to have a sequence in order to have an auto incrementing ID because it doesn't have anything built in that makes~~ the I ~~a new ID come when you are.

When you are adding ~~new when you're inserting ~~new values, new rows. That is the thing that we will try to mimic. We will create auto incrementing rows using something called the sequence. Here, sequences. The idea is that I want to put in several glossaries right into my tables. Here create a sequence sequence.

If not exists, ID SQL sequence start one. Okay, I will have this table called SQL, and [00:06:00] there I will have ID SQL sequence. I will copy this and I will create other sequences as well. I'll have one that is called Titan sequence. I will have one that is called Dtb sequence. Okay? They all ~~stuck ~~with one, copy this.

And create a new, here, I'll run this one. And you can see nothing returns, but it worked. And now we'll do similar to before, but now we'll go into from PG catalog. You saw that we had the PG catalog before PG sequences.

Okay, I put it in here and I'll run this one. And you can see here that we have our sequences, we have ID duct db, [00:07:00] sequence id, Python, sequence id, SQL sequence. And here you can see that the start value is one. Here's the mean. Value is one of all of them. And you can see here is the maximum value that it can take.

Yes. And it increments by one by default, right? That is what you need to know and how you can check them. Now it's time for us to actually create a table. Creating tables for each glossary for creating the glossary tables basically. ~~Glossary tables. ~~Here create table, if not exists.

I go into the database schema and inside there I'll create this SQL table. Okay. It's a little bit strange namings if you are like you think like SKL is, oh Yeah. It's the SKL script. And then we have a database, is the actual database, but however we use it as a [00:08:00] schema now. This is that you can practice in it depends on how.

Put the names, if you put it here create this table inside the schema table, then you should know that this is the schema, this is the table. Yes, I could have named it a little bit differently. I could have called it like db underscore schema or something and SL under table to make it even clearer.

But I will keep it like this that you get to practice a little bit of what we're actually creating. Okay. Let's create here. ID is integer, right? The ID is integer and it's default of next valve. Next valve will create the next value within a sequence. Our sequence is defined that it'll increment by one all the time, right?

Id SQL sequence. This means that. Whenever we're creating a new role, it will go into an incrementing, ~~this ~~this sequence. Okay? We'll [00:09:00] get 1, 2, 3, 4, 5, et cetera. Then we get our ID integer default value, and we can also make it into primary key.

**Kokchun Giang's video recording:** to make it into primary key, you just put it here. ~~Primary key. ~~Primary key means that it's unique and you cannot recreate the same using the same idea. Here we have word. It's a uniquely identifying a role. And for simplicity, we use. Auto incrementing integer for primary key.

We will talk more about how to design different keys like primary key natural keys. Surrogate key, et cetera. Later on when we come into our data modeling journey, that is the next course. This course is for analytics and to learn and navigate around in dtb and then and a scale.

Okay? We create the word we call it, we make it into string, and we'll have a description as well. And the [00:10:00] description is. A string as well. Okay, this is the data types. And basically this is what I want for creating my database SQL tables. But I'll copy this at two times because I will have some more tables here.

I will have database or the SQL table and we'll database duct DB table. And here I'll use the duct DB sequence. Okay. And otherwise it's basically the same. ~~And here we will have, not database instead of a programming. ~~Idea is that I will have in our database schema I will have several. I could have SKL duct db, I could have Postgres, I could my sq, l ms, sq l et cetera.

And ~~here in create table ~~here in programming, I can have Python. I could have a c, I could have Java. Rust, whatever, right? Here I will change this one to ID Python sequence. Otherwise, everything else is the same. I will copy this put [00:11:00] it here and run it. Yes, we have created this table. If I do desk, you can see that we have a few tables now, we'll have glossary here.

You can see there is. Schema. Database. Database and programming. And here, database duct DB database, SQL programming, Python. And you can see here is the column names and the corresponding column types. You can see integer var car. Var car, integer car. Var car. Var car is basically for string. Yes. Okay. Now we.

Created our tables. Note that this is not great in terms of crud. Crud, when you, we are talking about create, we're actually inserting data. Let's move on to our crud here. It's time to insert some data and to insert some data. It's actually quite we have seen that before. I will just make a copy into it.

I'll cop, because this takes a lot of time to write, I will copy it in [00:12:00] and I will explain what I have here. Insert into database data scale, and we'll have a word and a description. And basically it's here you have query. And here this is the word and this is the description, basically. What we do now is that if I copy this one in.

And cell here. And I copy this one in and I run this one and it has inserted one row. Now you can do from let's see from database dot sql, and you can run this one. And here you can see we get one automatically, right? Because of our sequence. If I copy in some more data now. Let's say I copied these things in crud and range conditions.

Okay? I copied these ones in and I will add a new cell here. Oh, actually that was stupid what I did now. Because I ran it now I will have two [00:13:00] of queries, that, that is an issue. I will actually just delete this one. Let me create a new actually I'll create a new cell here and I will delete this one.

~~Delete delete from,~~

**Kokchun Giang-3:** Delete from database. The scale where ID equals two, and I run this one and it has disappeared. What we can see now, if I run this one, you can see it has disappeared. Okay, great. Let's move on now. What I want to do is that I want to copy this code in here. I will create a new cell up here.

I copy this code in, insert into database sql, a word, common description, and I add a few more, right? I run this one and two here. And if I run this one, you can see ah. 1, 3, 4. And why is this three? Because I accidentally created one before, which was ID number two. It's auto incrementing.

The last ID that I ~~was incre ~~created ~~it ~~it was [00:14:00] two, now it's auto incremented to three and next one is four, and on and forth. Okay. This is normal behavior. What you could do if you want to have the. Old ID then you can actually change your sequence. It can, you can make the sequence to go back to another value, but I won't show you that here instead.

We'll move on now with the insert. We have inserted data into database sql, let's insert some data into. Python as well. Here is Python. It has value here. We can actually, this is a little bit too little, right? I will add I will add some more. Here we can do comma, and I would copy this one.

Let's see what we'll have. I can have type hint for example. Hint for which type in for example, parameters return [00:15:00] values, okay. In classes and functions. Okay, I can copy this one, can create some more, say, let's see, we can take topple for example. And here we say immutable work similar to lists. Also store records. Not just immutable list, that would be wrong. Okay. Or actually I don't want to auto format this. It's quite ugly. Okay, here we have for programming. I will copy this one in I will add a, I would add a cell here, paste that one in.

And actually here I will change my from database, from programming because if I just do from Python, you'll see that there will be an error because it's inside [00:16:00] of the programming schema, right? I need to do from programming. Dot Python. Okay, here you can see this is the data that I've gotten.

And you can see the ID has started from one because this is another id, another sequence, right? It's not the same sequence as we used before. And now I will copy one more thing from the for let's say this is for database attack db. Co copy this one in. Basically same idea here. I will create a new cell here, paste that in, run it, and I will do from database tb.

Okay. Yes. And you can see here's the TB data the glossary, right? Okay. That is that concludes the crate of crud. How here we have inserted some values, now we'll go into second one. We'll [00:17:00] read some values. To read some values, yes, we can do from if we do select star from.

Programming Python. This is equivalent to doing just from when it comes to Tech db. Both these will give the same results. If I do select Star from let me run this one. Select Star from Python. You can see we get this. It's the same and idea. Is that okay? If you want to read other ones, you take just from database dots, SQL, and then from database dot.

Let's see. Dot tp, right? This is same. That is the read. And from the, with the read, you can. Act you can do filtering as well. You don't need to just do select [00:18:00] star. You could, for example do~~ select star from, or yeah, ~~select star from database Dotb. Where ID is larger than.

Seven, for example. Then we'll get the last three. Yeah, exactly. We get the last three. Gloster risk that we have created. For example, you can do some filtering in your read as well. And I won't do much filtering here because we have already gone through filtering, but you should know that.

Okay. The read operation is using the select statement for that. And in D db there's a from~~ which ~~which is a shorthand for select star from. ~~Okay. ~~Okay. And then I'll go into crowd update here. Let's go into update. Here I want to [00:19:00] update I want to add a column called Learned. To to my database.

In order to do that we won't go into the update part first. Instead we'll go into the altar because now we're actually modifying this table, right? What I want to do is that I want to alter table database. DB and I want to add column learn. And this is a bullion and with default it to faults.

Okay,

Copy this one. What this does is that we. Al term means that we'll change this table. We modify it and we add this column [00:20:00] called learned, and it'll be a bullion and it has a default value of false. That is how it works. Okay? I will create. A new cell here and I will do this one. Right now if we do from database to Duck DB and then run this one and you can see ah, word description id, word description and learn and aren't is false.

Okay, great. Now I want to actually. I update this to true for those glossary that I have learned, right? That is the idea. And you can also do this for the other ones as well. I will just demo with one that you know how you are doing it. I don't want to take away all the fun from you.

Okay, now let's do. We would want to basically check or we want to yeah, we want to do check mark [00:21:00] for glossaries, which we have learned. For example, you have a front end where you have a glossaries here, and you can you do a check here if you have learned it or you maybe have a test and then you have scored correctly for the glossaries that you know.

And then it will give a check mark that you have learned it. Okay? And then this should be reflected in the database and if you want to reflect it in the database, you could do it in this way. Basically we want to do like this. Update database. Or actually we can do like this before we do this, is that we should select we should first do a check that ~~we ~~we get what~~ we are, ~~we want, right? Where Id in 3, 6, 7, for example. Say that I want to. See that I have gotten this correct [00:22:00] row. Let's go in here and see that.

Let's see. Yeah, you can see it doesn't work. That is good that it didn't work. Reference column I did not found in from class. Okay. ~~Yeah, because we, where ID in ~~Yeah, because we need to from, from, from database, ah, select, ah, i've done it wrong. Okay. ~~Select ~~select star I missed this from.

Okay. Yeah, that looks good that we did this check. Okay, I copy this one. Put it in here and now it should work. Yes, I get these three. The idea is that I want to change the learn tier, right? Now I do copy this and to copy quickly shift command shift option down button, or shift out down button if you're in Windows.

Instead of select now, we'll do update.

Update database. Actually ~~we update this one, right? ~~We update this table and we do set now. Set learned equals to true. [00:23:00] Where ID is this, okay, save this one or copy this one and paste it in here. Run it and you can see that. If we now run from database, do stack db and you can see that yeah, these three ~~are become, ~~have, become true because now we have learned them.

This was update. We modify the table. Let's move on to delete.

Delete here. Note that. Okay, alter, create and drop here. These are DDL, data definition, language. While these crowd operations that we're working on here, these are DML. Data or the read is actually DQL. The other ones are DML, data Manipulation Language. Okay. Yes.

They're a little bit [00:24:00] different. Now we'll go into deleting. Let's say that I want to select star from programming dot Python, where ID equals two, for example. I want to check ~~which ~~which we have here. Before I delete it. I add a cell here and I run this one and you can see, ah, type in, actually I don't want this glossary.

I think I written it wrongly, maybe? Yes. Then I check that which idea I have. Okay. ~~I will delete and delete from programming. ~~Basically delete. And then we have same as here, I copy this one. Okay, delete from program the Python where ID equals two. If I copy this one and I will add a new cell here, run it.

And you can see that if I do from programming and dos Python. [00:25:00] And run this one and you'll see Yes, the ID two has disappeared. Okay, great. And also you can do for example, we want to do let's just for fun let's do like this. I will go up here and I will find the, insert this one. I'll run it again.

Okay. Now when I have run it again, let's see what happens when I do from database Duck db. And you can see, ah, it copied right? 10 here or 11, it's same as duct. DB 12 is same as row two, et cetera. Then I want to delete all that are after 10. Okay, then we should select start from. Database DB [00:26:00] where ID is larger than ten two, just check that it will delete what we expected to delete.

I creates a cell here in between, you can see Yes, from 11 to 20. Perfect. Okay, I copy this one. And instead of select, I do select delete here. Delete. Okay, I copy this one and I can do delete here and run it. And if I do from database, that actually be, yes, it has disappeared. However, now when we create a new, we insert a new row, then it will be in ID 21.

Because last idea was 20, just to note that. Okay, these were this is DML still crowd operations. These are the crowd operations. However, we can drop the tables and schema directly using drop. And this is the DDL data definition [00:27:00] language. Here in drop, let's do this we can do drop table database dots.

Scale. ~~Actually I don't think I have something here, I'll just drop it. ~~And you can also ~~do like this yeah, we drop it ~~do like this. If I drop this one now you can see if I create a new cell and I will do desk. And you can see that glossary only has two tables, right? We delete those tables.

How about say~~ we want to create, ~~we want to delete. A schema containing all the tables. If we just do like this, if I do drop schema let's see a schema programming, see what happens if I drop this schema. I'll create a cell here and you can see there's an error. Dependency error cannot drop entry programming because there are [00:28:00] entries that depend on it.

Table python depends on schema programming, right? Use drop cascade to drop all dependencies. It could have a lot of tables inside of this programming schema. If I want to delete them all, I do drop schema programming cascade. Then I don't need to go in and drop each table, right? If I just add this cascade here and I run it and it's fine.

If I do desk here, now you can see there's only for duct DB left. What you can do to check this is also like from informations EMA tables. And you can see these are the tables that exist and you can do from information schema and dot schema to see which schema there are. Now you can see there's only this database schema and this [00:29:00] main schema here, which I don't have anything.

Yes, now remember to. Commit. Actually I will close this down. Control C and or control B actually, and now you can see the connection to the duct B is lost. And you can see there's still a do valve file here, and I will delete that one for the lib B glossary db. Okay. ~~And ah, it says that it's running right.~~

I'll close this one. ~~I will do TB glossary db it's still running. ~~There's an error here. Let me let me debug this.

**Kokchun Giang-4:** Okay. I'm actually quite glad that I got this error. You might think oh, I'm strange that, I'm glad that I got an error or a ~~bag ~~bug, but this made me understand why I got, I've gotten it now. ~~How did I. ~~I'll show you a little bit how I did for debugging. Basically you saw that I couldn't open my glossary db, right?

There was some strange error. I deleted my dot valve file to see ~~if if something ~~[00:30:00] if I could open it. And when I open it, we saw that there's no changes, of course, because the write ahead log ~~hasn't it ~~hasn't been passed. Correctly. What I did was to, I deleted the database and I restarted again, and then what I did was to run through each cell in the notebook to see whenever I get an issue.

~~And ~~and what I suspected was~~ when I, either, ~~when I dropped the tables or when I did the altar I checked for the dropping and I checked for the ~~altar ~~and I saw that okay, it was actually the ~~altar ~~that was the issue. ~~Okay? ~~Whenever we did this code here. ~~Alta ~~Table database, ddb.

There is an issue because before I did that and I closed down the local UI ~~and ~~and then I opened up the D db CLI and everything works. But after I did this ~~al ~~Alter table. Inside the local ui. I [00:31:00] saw that okay, this error appears. Okay, good. Then I isolated that it's during the altar,~~ then I saw okay.~~

Could it be that because I use these strange names, database, tech db, that when I. Alter this one. It will make it not possible to parse or some strange thing. Yes. And then I tried to alter the programming dot Python, that schema and that table and did the same altering. And yes, it didn't work either.

It became the same error when I did it in the local ui. Okay. And then I try to, okay how if I don't do it in local UI and do it directly in the CLI. Okay, if you have local UI open and you do it in the CLI, then it won't work. It'll be the same error. However, if I open it up in. And the normal CLI without local ui, like this activity glossary and I do this altar [00:32:00] and the error doesn't appear.

It seems to be like there seems to be a bug in local UI together with this, alter. I'm not sure exactly why but this is what I have noticed. To work around this I will show you from start how I would do I will first close this down. I will remove this one, and now I will do T db.

UI glossary, reductive B. And you can see this is the code that we have created. I will run this one. I'll run this one. I'll run this one~~ run. I'll do shift enter to run it, shift enter shift, enter. ~~And here. Yeah. ~~Shift enter, shift, enter, shift, enter.~~

And here I will do, comment this out. ~~This is the dangerous code and this was the code I tried. ~~This is also dangerous. If you want to use that one, you should also comment it out ~~that ~~and here I will say a note here, note there will be a bag here run. [00:33:00] Run this ~~in ~~in the normal D that P-B-C-L-I and it will work.

Okay. What I do here is that I move back to my terminal. I do control D to close it down and you can see do valve. This doesn't disappear. With the duck DB glossary, duck db, and you can see if I do disc, I have my tag tables, everything works. And I just copy in this code here, copy it in, and now I do.

Desk again. And now you can see the learned column is there. And now ~~I will, ~~you can see there's not valve file here. If I do control D, it disappear. Everything works very smoothly in the CLI. But the local ui, there ~~is ~~some bugs. But we will open up the local UI [00:34:00] again, and here I will move. After the here is the alters, right?

After that ~~one ~~we'll continue to run it run. And everything else will work, I think. Okay, now I have run everything. There should be a run all button. I haven't found it or they haven't implemented it yet. ~~Here. ~~Now I will just do control D and then there's this bag again that the dot valve file doesn't disappear.

What we need to do is to open it up with the normal CLI and here. What you can do is you can see desk, ah, everything worked. We have deleted the other ones, right? Because that was what we did in the ~~later ~~later stage of the local ui. Now if I do control D, you can see dot valve file disappears and it everything works as expected.

This is the workaround. You need to run this inside the ducty BCLI and not inside local ui. ~~Okay? ~~Okay, now make sure [00:35:00] to commit and push your changes to GitHub and also ~~mark this ~~write this note about it. That you can work with it. Yes.

Okay, in this video we have gone through crowd operations. We started out with some~~ we have combined that crowd together with some d dls, ~~DDL data definition, language, such as create, alter, and drop. Okay. And also we use, ~~then we did the crowd operations. ~~Create, read, update, delete ~~create ~~using insert read using select, and also wear class and update using update and delete, using delete.

And then we found out there was a bug in the local UI with the Alta statement. What we did was to the work around was to run it ~~in the. ~~In the CLI in ivb. And afterwards go back ~~to, to the ~~to the local ui. I hope that you learned a lot in this video and that you get more and more familiar with DDB and it's power.

This is super fun and I hope that I will see you in the next video ~~and ~~where we'll go into grouping of data. That would be fun. [00:36:00] Okay, see you then. Bye.

