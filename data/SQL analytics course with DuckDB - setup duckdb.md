# SQL analytics course with DuckDB - setup duckdb

[00:00:00] Hello and welcome to this video where we'll go into setting up DDB for both Mac and for Windows, that no matter what the system you're using, you will be able to set it up. And also I'll show you ~~in ~~where ~~you've. ~~Find the downloads for duct DB that there's instruction for Linux as well if you are ~~in ~~Linux.

Yes, and for this I will also, after we have installed it, ~~we'll try ~~try it out with duct DB version to see and also duct DB CLI to see that it works in the terminal. And then afterwards I will also ~~use. Will ~~ingest some data and show that it's possible to query the data both in Windows and in Mac.

And then finally we'll go into DDB local ui, which is~~ a ~~new since~~ summer 2024. They released the duct db. Or 2025, actually, ~~summer 2025, they released the DDB local ui, which is really nice way to work ~~with the ~~with ddb that you will get the feel of how ~~that ~~looks ~~like.~~

This is part of the setup as well. It's quite easy [00:01:00] when you actually ~~have ~~have it installed. Let's move on to~~ we ~~start with Mac and then we'll go on to Windows.

**Kokchun Giang-2:** starting in the browser, we go into a DDB installation here. And you can see there is different types of platforms. You just choose which platform you have. If you have Linux, you click on here and you have this call command to copy. And for Mac, there is another call, comment to copy. However, for Mac I will actually use brew.

You can see here in alternative down installation methods. There's brew installed duct db, I'll use that. However, make sure that you have brew installed on your computer. Just have here you can see this is my terminal, and here if I write ~~brew. ~~Brew. You can see BREW Works in my computer, but if you don't have brew ~~in your com, ~~in your Mac, then you should search for it.

Home brew install. Okay, and then you go into here. [00:02:00] This is ~~the ~~for installing the Home Brew. It's a package manager for Mac, which is quite nice~~ which ~~you can use this to install Duct DB and many more softwares. Okay, let's go back to terminal. You just copy and paste ~~this ~~this command here to get home brew if you don't already have it.

But if you have it like me, then we'll just do this brew install, db. Okay? It'll install duct DB in the system.

Yes, and it's done. Now we can see duck db dash version, and you can see it's duck DB version 1.4 0.1. Okay. This will work. We will do control T and I'm outside of duct db. Okay, great. Let's move on to windows.

**Kokchun Giang-3:** here I am in Windows, and you can see that we have [00:03:00] ddb installation, the same page as before, but the platform here, you choose Windows. Okay? And now you take direct download. Download this one.

Okay. To download it, and then we go into, downloads and you have this compressed folder here. Double click it. Actually we go back a step. I want to right click and extract here. Extract all extract. And you can see this is my folder and this is my duck db. I'll copy it, and I will go into my.

Into this PC here and I will go into C Drive. I will go into program files. Inside of program files. I create a new folder called D DB [00:04:00] and I go into this program, files. I paste it in. Okay, now you can see our duct DB is inside here. ~~But let's go into ~~and now if ~~you take you open up a command ~~you open up a CND or ~~power, a ~~PowerShell, ~~actually we can take.~~

Actually, it's best if you would use Git Bash because there ~~you will have all the, ~~you will have more Unix commands, and you'll have more bash commands, which is quite good. However I show, because I'm using a virtual machine now and ~~I don't have a ~~I don't have GI passion installed, I'll use PowerShell and it works as well.

Now if I write Duck db. You can see that~~ it doesn't find it. I think. Yeah, ~~it doesn't find it. And that is as expected because we don't have the path to this duct DB file. Then we need to set this path. You're going to search here and you ~~type in, let's see, what should we type in? We should ~~type in environment variables, edit environment variables for your account.

And then here. [00:05:00] You go into this system, variables here, path and you click edit and you come into here, edit system variables, and you create a new one and browse. Here we need to find where program files is. Let's go down to find this PC C. And we have program files here. And we have duck db, right?

Okay, now you see it's pointing towards duck db~~ okay. ~~Okay. And we restart our terminal or our PowerShell. Type in PowerShell again. We have PowerShell ~~duct DB version,~~

duct db. Okay. It doesn't work. Let me think. Okay.

**Kokchun Giang-4:** to find out the issue is you go back to let's see, yeah, program files. You can see Duck [00:06:00] db. And first, when I double clicked it it didn't work. Now it works. You can see that ~~we have, ~~we get the ~~duct ~~DB up here, but before it said there was an error that it was missing.

M-S-V-C-P 140 DLL. basically this is a DLL file, which you get ~~when you are in, ~~when you install~~ visual, ~~Microsoft Visual c ~~plus ~~REDISTRIBUTABLE packages. ~~Okay? ~~Because this is what~~ Dr. ~~B depends on, ~~what we, ~~what I needed to do was to actually go in and search for that. ~~I sat, ~~I went in here and I basically, ~~I ~~searched for Microsoft Visual, c ~~plus ~~Redistributable, latest supported downloads, and then download this one and then installed it ~~basically.~~

And after I've installed it, I go into PowerShell~~ power Shell, ~~and we can write Duck DB version and you can see it works. ~~Okay. Control C Duck db let's see. ~~Duck db. Dash, dash version. Yeah, exactly. Here you get the version out here? [00:07:00] Yes. Cool. This works. In Windows. But note that ~~what I said before, ~~if it worked directly after you just have pointed the P, then you don't need to do anything more.

But if it doesn't work and you get the error that I've gotten, which was Ms. VCP 140 ll missing. ~~Then ~~make sure that you install the Microsoft Visual c plus redistributable packages. ~~Okay? Okay. ~~Now I'll move on back to Mac, and then what I show there will be the same for Mac and Windows. Just keep watching even if you are ~~in ~~in Windows.

Now it's time for us to ingest some data and test out the local UI as well.

**Kokchun Giang-5:** I'm in Visual Studio Code now, and basically what I have is just a CSV file of aging near YouTube 2024 to 2025. I just picked out the last 365 days and exported it out. ~~Some some videos basically. ~~If you look into this one, this is how it looks like. It's a CSV file, commas, ed values.

Basically we have a few, video [00:08:00] titles and number of views, et cetera. And then number of subscribers. What I want to do is that I want to ingest this data into a duct DB database to see that it works and show it in local ui. Okay? To start with that, ~~let's just I'll make a new, ~~I'll use the terminal here.

~~This ~~you can download this data set from ~~my ~~my repository, which I will leave in the link in the description of course.

Let's create this file structure here. ~~Make this, ~~create a few folders, create data, and I'll create SQL.

This creates two folders. And then I will move this engineer into data slash engineer data slash yeah, I will copy this. ~~Ah. Let's yeah. ~~Engineer YouTube. 2024 2025 csv. ~~Okay. I move it in ~~basically you can see it here in the data. Okay, good. And then ~~I will also create a few, ~~I will go C, the SQL, and then I will do touch ingest data [00:09:00] sql ~~basically~~

~~.~~

**Kokchun Giang-5:** You can see here, there's this ingest data, sql, and here the idea is that ~~I want to ~~I want to read this data from the engineer YouTube 2024 2025 CSV into DB directly. We'll write an ingestion script here. Basically let me create table. If not exists, I'll create this video table or videos as, here we have a Subquery Select star from Read CSV Auto.

And here I will put in my. Fine name, basically. Copy this one, paste it in here. [00:10:00] And ~~I want to be, I want, I need to do this I want ~~I want the db database to be outside of both these folders. What I need to do is to actually reference it using data slash I'll show you why. This is why.

I go out cd dot. The idea is that now ~~I am in, in, ~~I am in the root folder and in the root folder I want my D DB database file to be in. From the root folder, ~~if I go in, ~~if I run this ingest data, sql. Then relative to the root folder we go, we find this CSV file in the in data. We need to go into data slash the CSV file, right?

That is the reason why I need to have data here. Okay, what does this do? Create table, if not exists videos has select star from. This read, CSV. It will read the CSV and then it'll [00:11:00] create a table for us. Just like that. Okay? And then if, when we have this table, then it's quite easy that we can just query this table, right?

What we need to do now is to create our database, d db. I call it YouTube videos dot db like this. However now I want this SKL script to be run. In order to run this one, I need to do something called input redirection. ~~Actually other direction? ~~Yes. Now we go into SQL slash ingest data.

It means that this SQL code will be run inside of this T DB CLI. I run this one, and now you can see we have this YouTube video ~~db. If I do duck ~~[00:12:00] db. YouTube videos, active B, and you can see this is my Dr. B. I will clear the terminal and we can do Describe, and you can see, ah, YouTube videos is here.

Good. The database name is called YouTube videos. Schema is in Maine because we haven't specified otherwise. And the name the table name is videos. And here you can see column names and column types, et cetera. Okay, now what we can do is to just select something here. Select star from let's see, videos.

And you can see, yes. Here are my videos. Here is the video title and here's the subscribers, et cetera. And also since it's db we can do, we don't need to do Select Star from, we can do just from videos and it works equally. Cool. Yes. This is quite cool. I will dol D in windows [00:13:00] is control CI think.

To jump out of the ddb. Okay, now let's do tdb Dash ui YouTube videos to tdb. I'll open up the local ui. Okay. It'll automatically open up a Windows, let me see.

**Kokchun Giang-7:** here is the Duct DB local ui. Let's look into this wonderful thing here. Let's start here. Here are the different notebooks. I have a few notebooks since before. However, you might not have any notebooks if you just installed it. Okay, since we ran DDB dash ui and then we chose YouTube videos, right?

We can see here is the attached database. This attached database we have main, and then we have videos and you can see here's a little bit of summary. On the videos on the, on ~~how ~~how this [00:14:00] table looked like. For example, here you can see there's 263 entries. Here's how many entries in this one and how many entries in this one, et cetera.

And the length, you can see there's a histogram here. You can click it up and you can see a little bit more clearly here. Already here you can get a lot of metrics quite quickly already.

However, let's dive into analyzing our data very simplified. Create a new notebook. We'll call this ~~engineer, ~~engineer YouTube channel analytics, for example.

~~We create an, ~~this is similar to Jupyter Notebook, however, for Duct DB and for S scale. Here you can add a new cell, and this cell you can see which. Database it's connected to. If you have several, you can see, you can choose another one. But this one is connected to here. Let's run desk to describe, to see what we have.

Okay. And here [00:15:00] directly you can see there's a lot of underscore duct db ui. These are needed for the duct DB ui. Don't mind them. Instead, here is the database that we want YouTube videos, right? And here we have the main schema. Schema is the logical grouping for our tables, and here is the table name.

We have videos, since we're already connected to this database, what we all need to do is that we can do select start ~~from you ~~from videos. And then we can run it. Let's see select like this. Okay. And I run it with shift enter, and then it'll jump into the next cell and run this one. If I run it with control, enter, then I will run this directly or command enter without jumping to the next.

Okay. As you can see here, we have. We got all our columns, we get our data, and you can see here is summary statistics for that. Similarly, we [00:16:00] can do like from videos because in ivb you can use from videos instead of select star from, it's similar, this gives the same results. And here you can see we don't want this first row here because this is the total.

Or maybe you want that May, but I don't want that. I would just select star from videos offset one and I don't get the total. Okay, great. And let's see here. And ~~we can do, ~~we can filter it a little bit we can do select start from videos. Where let's see where this thing up is larger than, for example, 300 and I'll run this one and I will [00:17:00] offset one as well.

Offset one. And here you can also make it a little bit cleaner by cleaning up, like you can you can indent it as well. Okay, here you can see we have filtered it out based on a number of Disney air views. And you can see here's the. We can do like this, exclude in the hole as well. And let's see, in the hole.

Yeah, I don't get the hash the IDs. Okay, great. And here you can see the videos. ~~I, ~~you don't need to ~~ex ~~exactly understand all the s ~~scale ~~code right now. But what you need to know is that this is a user interface. You can work with it, ~~you can. ~~You can easily work with SQL, work with DDB and interact with your database.

What I suggest is that a good [00:18:00] thing to do is to ~~copy and paste we ~~copy and paste all our queries. Describe, maybe I don't need to copy, but I can copy this one. I can go into. Here, my ~~SGLI ~~can create a new file that's called Queries ~~sgl. ~~And I say this one, I'll go back here and I will copy yeah, this one maybe or this one I will take.

Here you can see there are several. I also did auto formatting here. Like this. Okay, ~~what, ~~these are a few queries that you can save locally. And why do we do this? Because ~~yes, ~~in Duct DB local ui, it is stored locally. However, you can only go into it with your own computer. When you do like this you save the SQL scripts and then ~~you have you, ~~you also put it into your ~~GI ~~Git repository you can version control it.

Make sure that ~~you can com You then ~~you can. Add it, you can commit it and then you can [00:19:00] push it up to GitHub. In this way, you version control your SQL code as well, and it's a very good practice to do. In this way you will be able to share the code. You will have a backup and you will be able to go back and see ~~whenever things go ~~whenever you need to access the code.

Yes. That is good practice to do. I really. Hope that you are doing that.

Yes, this was super cool. We managed to set up DDB in both Mac and in Windows. And I also showed how to do it in Linux, or I showed you where you can find instructions for that. ~~You, ~~we started going into the DDB downloads page, and then ~~we, ~~for Mac, we did brew install DTB.

And then for Windows it was a little bit trickier. We downloaded the. DDBC xe. And then the D DB xe, we extracted it because we had a zip file and then we copied it into program files. And then we ~~did ~~edited our [00:20:00] environment variable that it'll point to that program files. And then finally what we did was ~~to I, ~~I got an error because I didn't have that DLL, what I needed to do is to install.

Redistribute the Microsoft Visual Studio c plus redistribute redistribution. After I installed that, it worked okay. And then afterwards I went back to Mac to show you ~~the ~~how to ingest data. We ingested the SQL. Some CSV file with with something called input redirection.

That ingested data into the D DB database file. And then we showed it in the CLI command in the C-L-I-D-D-B-C-L. And also finally we went into the dtb local ui, to see how it looked like, and~~ and then ~~did a few analytics queries. Not many, but this is really fun and I hope that you've learned a lot and have a set up dtb because.

Both. It's very easy [00:21:00] to set up and it's very easy to work with. It'll make SQL fun for you, I promise you that. I hope that you learned a lot and see you in the next video. Bye. 

