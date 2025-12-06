# SQL analytics course with DuckDB - views tutorial

[00:00:00] Hello and welcome to this video on views. ~~And ~~before we have worked with DD BSQL quite a lot, and now we have come into views before we worked with tables. There's a difference between views and tables.

View is basically a saved SQL query as a virtual table. It's a. Virtual tables.

What does that mean? It means that the query is executed every time it is accessed. The data is not stored. , Why do we want this? The purpose of views is to restrict access to underlying tables. We want to simplify complex queries ~~for examples involving joins of many tables. ~~For users

we customize data for different users. That is possible when we have views, right? And we can maintain stable interface between underlying tables and downstream consumers. For [00:01:00] example, when column names changes in the underlying table. We still present the view as it is~~ to the co ~~to the consumers.

While this might sound a little bit abstract, let's go into the coding part. I will jump directly to Visual Studio Code and we have some data that has been generated with LM housing sales data, and. Neighborhoods data. Here in this video, ~~we will, ~~we won't actually restrict the access of the data, but in future videos you will learn more about access control but know about views ~~we'll ~~will be helpful for restricting the data.

. But let's move on to the code now.

**Kokchun Giang-1:** I start out in this visual studio code, and you can see I have the data here, I have housing sales. Let's take a look into preview. And here you can see sale id, neighborhood address, date of sale, et cetera. , This is the [00:02:00] housing sales, and then I have the neighborhoods CSVs. Open that in preview and you can see the neighborhood, the name, city Postal Code, the median annual salary should be right.

And let's see, yeah, income, SEK. And here we have population. . Yes. Let's ingest this data. Make the SK, L and CD into SQL. And here I will touch, ingest data skl and I will touch on views. SK. You can see these two. I'll go into ingest data. This is basically what I've done in all other videos or many videos.

Create schema, if not exists. We want to have the staging schema because this is the landing zone of our data, and then we create table, if not exists, staging dot ~~housing ~~[00:03:00] housing sales. As, let's see, ~~select. Not schema ~~select star from read CSV auto data ~~slash ~~slash housing sales csv. And here I will copy this one with ~~SHIELD F Shield ~~shift option down button, or shift out down button if you are in Windows.

Let's see. Housing sales. And then we take the, actually I can do this. If I mark this one, I can do command D. You can see it marks this one and Command D. It marks the next one here. ~~And instead of, ~~and here I will just write ~~neighbors ~~neighborhoods. And you can see the changes on both these. Some hotkey magic.

That is super cool and shift option F to auto format it like this a little bit. Nice. , Now let's create this data. ~~DD ~~I need to jump out ddb data slash [00:04:00] housing. Do DDB and input redirection, SQL ingest. Data. , Now I have this one. I will do duck db dash UI data slash housing db and I'll open up the ui.

**Kokchun Giang-2:** , let's start with creating a ~~view. Create ~~view. I would put this view inside of staging. Similarly as with tables, we place it in different schemas, right? Staging dot average price type. This is the view that ~~I call ~~I call the view this. As select. Property, like select from staging housing sales.

~~And here I will do group by. ~~If you take a look into housing sales, we can take from staging dot housing sales, take a look into this one [00:05:00] and you can see there's actually I will do desk table. Staging, housing, sales, and you can see here we have sale id, neighbor id address, date of sale, property of type, et cetera.

I will group by the property of type, group by property of ~~property ~~type. . And here we create this view as this select statement. What do I want to select from this ~~world? ~~I want to select the property type, and I want to select average of sale price USD. You can see here we have sale price, USDI calculate the average from it.

As average price USD. . This is a view based on this table housing sales. ~~Take a look into this now. ~~Right now you can see zero rows [00:06:00] returned. Yes. Zero has returned because. ~~We, ~~we don't put in any data here. We have just created this view that is ~~this ~~this SQL code. If you want to query from the view, we want to actually see the data and it will execute the ~~view, execute ~~SQL queries there.

Query the view and you do it simply by, as normal from staging average. Price type. I will copy this one and it works as you can see. And now we can see we get the property type and the average price. USD. Yeah, I could make it a little bit nicer of course, but yeah, that is fine. ~~Now we could, yeah, we could make it a little bit nicer.~~

Make it into integer. Integer. Let's see. Then it's this one, right? Type cost it into integer. Yeah, let's see. View with name. Ah, right. It already [00:07:00] exists. Create or replace view. Now it replace this one and you can see, yeah. Now it's a integer that, . Nice. . Let's continue. Now, I want to modify an existing view and to modify it.

You just saw me do it like I did the replace. I'll just copy here, copy this one and move it down. Here you can see we create or replace. View staging average price type as select property type, average sale, sale price, USD, and I will also add another one. I will add for example let's see, comma max of date of sale as latest [00:08:00] sales.

. Here because the date of sale you can see it's a date. If you take from let's say, select date of sale, comma, for example sale. Sale ID for example, from. Housing staging, housing sales. If I run this one, you can see there's, there might be more than one for one object. There might be more than one date because it might have been sold several times, right?

A house could be changing owner 2, 3, 4, 5 times. And you do take the max of it and you get the latest sale. , Here we do group by property type, and then I can order by property type as [00:09:00] well. , Now I have modified this view, actually I can make it in between here. And make another cell in between and run this one.

And you can see zero rows have returned. And if I run this one now, you can see it's different because we have updated our view. Now ~~it's, ~~you can see the latest date here. ~~And, ~~and what property type and average price. USD here. . We can also list the created views. Basically from information schema, you know that we have gone, in previous videos, we have used the information ~~schema, ~~schema data to see which ~~s ~~schemas we have.

We have used information schema, the tables to see which tables we have. Now we have information schema views. And where table schema equals to staging, we can find out all those that corresponds to the staging. I run this one, and now you can see it's [00:10:00] from the only, the ~~housing ~~housing staging a price type.

This is the only view that we have. You can see the view definition here. It's the view definition actually quite small here. But the thing is you see it's the SGL code. And afterwards we have let's see, yeah, this is what you need to know right now. ?

Here ~~we can. ~~We can we can drop a view if we want. ~~Drop, ~~here is a list of views in a schema. ~~Drop a view we can drop this view. ~~Drop view staging, do average price type I can copy this one. Add a cell, run it. And if I run this one again, now you can see that we don't have any views.

Let us create another view here. This one is I'll make a ~~simple view. ~~Simple view from multiple tables. ? Create view staging [00:11:00] dot average. Price hoods. This is quite hard to spell. ~~S select from What do I want to ~~select from staging dot housing? Sales. ~~SH ~~left. Join Staging Neighbor hoods.

And I will use n as alias here on H Neighborhoods neighborhood. ID equals to~~ h Let, let me just make it a little bit clearer on ~~H neighborhood Id. . And then we want to group by n name and order by, I'll explain this soon. CN name. ~~. This was quite long query. We have max of NCC.~~

We have. n.name, we'll have average of H sale price, USDS, average price. . What does this do? Well, let's [00:12:00] create this one first you can see nothing. And then we need to ~~do ~~from staging average price neighborhoods, because this is a view, you need to actually. Query from it in order to get the data.

~~. Or actually you need it. Yeah, you do need to do that. ~~What we have done here is that, ~~, you, ~~we start from~~ we take the ~~staging housing sales, we can take a look into housing sale from staging housing sales. And you can see here we have Ali, ~~neighbor, ~~neighborhood id, et cetera. This one is the first one I select from, and then a left join the staging, ~~not ~~neighborhoods in.

I left join on, we can take a look into from staging neighborhoods, ~~hoods, ~~and let's see. And here you can see the name, you can see neighborhood id, right? Here's different neighborhoods. We can see the median annual income and population, et [00:13:00] cetera. . But we have the neighborhood ID that was corresponding in both that we could be able to do .

Neighborhood ID here is used as the key ~~that we, ~~the column that we join upon. We join on this. And then we group by the neighborhood name that we get each neighborhood name here and we order by city and, ~~and, and the ~~name. And then what we do is that ~~we take we, ~~we aggregate, we use aggregation function of max to end city basically.

And ~~we will, ~~if it's the same city, we will ~~get we'll get one, we'll ~~get one of them. But if it's different, it's actually keeping ~~the, the one that is the, the, you will get ~~one of these two. Here, and then I'll pick the name and I do this average. Making a aggregation function here for the sale price in USD and average them.

~~It's a little bit strange actually, that we get ~~it's actually not strange

. . I actually fixed it. Here there was a little ~~bug, bit ~~bug. I chose~~ here h ~~neighborhood ID instead. It would be ~~like, it would ~~joining on itself, of course, then it would be wrong [00:14:00] in doing average price because we're averaging pricing on the same. But now I changed it to h and n here and then.

And the result is correct here. , With this, make sure to~~ we ~~close this down, SORL D and you can see there's no, there's a dot valve file here. . What we should do is attack DB data slash. What is this called? It's called housing right? Do db and then I will do Control D to close it down.

And now you can see the DOT valve file disappeared. Yes. And now you should commit and push to your Git, GitHub repository ~~and ~~and finish.

Yes, in this video we've worked with views in D db. We have learned a little bit of concepts about views, and then we have implemented the views and you have seen it in action. And we have showed where we find the views. We have dropped views ~~and ~~and basically [00:15:00] used it~~ to, ~~to combine several tables to make a simple interface for the end user. I hope that you've learned a lot in this video and I see you in the next video. Thank you. Bye.

