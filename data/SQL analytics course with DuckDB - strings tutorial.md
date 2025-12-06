# SQL analytics course with DuckDB - strings tutorial

[00:00:00] Hello and welcome to this lecture where we'll go into strings and working with text in ~~d ~~db. In the previous lecture we have gone through the concepts and ~~now we're going, ~~now we're ready for the coding part.

And here we'll use a very. Unclean glossary CSV file, which you can look into here.

As you can see, there's a lot of spaces in different places. There are a lot of characters ~~that have been, some ~~that have been uppercase and lowercase, et cetera. It's a really messy who have done this mess. But it's time to clean it up using~~ string and ~~string functions ~~in ~~in duct db.

We'll start with ingesting this into our duct DB database, and then afterwards we'll use different type of string functions to clean it and also show. Slicing ~~show ~~some pattern matching, et cetera. That will be very interesting. Let's move on to Visual Studio Code. ~~And, but ~~if you want this ~~data set ~~same data set, [00:01:00] you should take it from my repository.

Which I will link in the description. Let's move on to Visual Studio Code.

**Kokchun Giang-1:** Here I am in initial studio code, and you can see this dataset. This is s QL terms csv, and you can see, ah, there's a lot of spaces here, and here's capital letters, here's lowercase, here's lowercase with capital, here's capital, et cetera. This is really messy data and you can see there's a lot of spaces here, et cetera.

Let's ingest this one. And how to do that? We need to create, make the SKL folder and then touch SL oh, actually go in cd, SKL 'cause I will touch all my scripts at once. Touch ingest gloss, glossary, scale clean glossary. SK. And then I will show [00:02:00] you some other string functions and methods.

We'll do pattern matching

skl we'll have string. Functions or text functions, SS scale, and then we'll have slicing s scale. ~~, S ~~you can see there's a lot, and I can see here there's a lot of scripts now. Let's move directly to ingest glossary here. We will. We will actually divide it into s schemas. Now, if you've seen the concepts part of the strings we will create the schema here.

Schema, if not exists staging. This staging schema, I will, this is the landing zone of the data. Here I, we will put in the data from the raw data and then we'll have a refine layer where we'll do the transformation of the [00:03:00] data. ? Create table, if not exists. This is the landing zone, ?

We go into staging schema dot. Let's see what it's called. ~~Yeah. ~~S Scale Glossary as ~~select ~~select star from Read CSV Auto Data slash SGL terms. Csv. , This is a good starting point. Let's see. Yes. Now we can see that we create this table inside of staging called SQL Glossary as this select statement.

It means that we will get the data into our DDB database, but we need to do input redirection for that. I will do cd.to jump [00:04:00] out and then I will do duck DB and let's see what this should be called. I would call it like strings duck DB or glossary. Duck db. Yeah, I call it glossary or strings.

Strings start and input redirection. I will go into SKL. I will go into. Ingest glossary. , Good. And now there's the strings start B. The duct B dash UI string.

**Kokchun Giang-2:** let us start with string functions. Text functions. I'm in dark DB documentations, and here you can see text functions. And here there are a lot of functions here. This is very comprehensive. Just find the ones that you like. And and which works for your particular case.

I will go back to the local ui, which I opened up here, and let's go into string [00:05:00] functions or text functions, ? We will see if we start with here, I can do from staging, because if I do. S scale glossary, you can see that it won't work because this it goes into main by default.

Do you mean staging not s scale glossary? Yes. I do mean that staging not s scale glossary. And you can see that it looks really strange, ? Here, there's a lot of strange things. , Let's us do for example, I want to. I want to trim SGLI want to remove leading and trailing spaces, remove leading and trailing spaces.

For doing that, we can do select stream s scale. We have sq word, ? Sq word. What do I want to stream? I want to stream a space, ? Streamed [00:06:00] word. And then I will have yeah, let's start with that from. Staging dot skel glossary. Copy this one and new cell here. Run it and you can see what happens.

We have TriMed, all the spaces before and after, ? Now we can do, I can continue here and for example, I can, use this streamed word now streamed word and I can pick out for example, the first character here as first character, and I can copy this one streamed word of minus one as last character.

.

**Kokchun Giang-2:** , You can see first and last character. Yes. Why do you want to do [00:07:00] that? I'm not sure, but we can do that. Note that we have done this function, the stream function to the SKL word and we have given it an alias. And after we've given it an alias, we can use this column to. Pick out using the indexing, ~~the ~~the bracket operator use it as indexing to get the characters we want.

We can also do slicing, I'll show you slicing later on.

Here we can continue, we can transform character to uppercase. Let me just copy this one again and I will do. Into upper here.

I can do, yeah, actually I can yeah, I can just copy it in this whole thing. I want to transform character to uppercase, and here I'll do upper.

You can see now we're nesting [00:08:00] the functions.

If I open a new cell here and run it, you can see now we have uppercase of ~~all the, of the, ~~all the trim words. It's called trim word now, but I could call like upper upper. Word, for example, or I can just call it a scale word that also work, a scale word or a scaled

copy this one. Paste it here. Let's see.

~~All , ~~because I need, I used the here trim word, I need to have S scale. Term, I guess I used a scale term there. A scale term and now it should work. Yeah, exactly.

Yes. We transform the character to uppercase. ~~We can replace you can see there's if I will show you here. Actually, I can do a new one. ~~From staging [00:09:00] dot sq glossary, you can see the description here. There is a lot of, there is like a lot of spaces here, ? This is more than two spaces, here is more than two spaces, et cetera, and there's irregular number of spaces.

But let us try to remove this. We'll try to replace two or more spaces with one space. Select description, comma, replace description, comma two, space with one space. . S cleaned. Description from staging, s [00:10:00] glossary?

Let's see. I think there might be some issue. Par error. Unexpected one at first. Ah, as first character. Yeah. . Actually it like this format, it doesn't understand D DB syntax. This is D DB syntax and not in other SKL dialects. ? But this it works. I copy this one and please put it here.

And you can see now the spaces have become. Somewhat cleaned, but here there's still some issues. We replaced, ~~we, ~~we didn't do two or more spaces. We replaced two spaces with one space. .

We will show you later on how you can clean this using regular expression. But now I will keep it as as this is we have RegX and pattern matching in another [00:11:00] script. Let us try also to concatenate strings.

Concatenate strings and to do concatenation you have select concat. Or actually, let me,

let me copy. I can copy this one here.

And I will remove these. , We have trim and we will use upper on the trim as well.

Let's see,

upper with the trim as,

as the command. And I will use like this. I will do also, I will do concat comma command like this. This means that I have, if you look into [00:12:00] here, let's say there's an error because there's an. Ah, command needs to have a ~~parent ~~ending parenthesis here. . Ending parenthesis there. Yeah, it works. And you can see there is a command word after each of these words.

Not entirely correct, but we have command afterward with the space here because we concatenate the first. The first column the first word with the second one using the concat. Similarly, we can do that, but ~~instead of concat, we use, ~~instead of concat, we use, let's see here,

~~to ~~to vertical bars.

And you can see it works the same because two vertical bars means concat ~~in ~~in duct db. That is quite cool. We can, I will do a commenter con concatenate [00:13:00] strings with vertical bars. . We can also do we can. Extract Substrings. Let's see. I'll copy this one. The first one, paste it here.

Now in this two, ~~and here we have, , ~~we have trim S scale word. ~~And I want to let's see. Trim S scale word. I want to do, ~~I want to do a Substring now, extract Substrings.

Actually, I will remove this one, trim it here. Let's see what we get here first. Run it and you can see yeah, it has TriMed because it'll trim by space by default. Anyway, now we'll extract ~~some ~~some substream from this. I will actually keep this one that you can see the original one.

And here I will do x sub strings. [00:14:00] One five as extracted like a first five characters.

~~Let's see. ~~Scaler function with subs. Strings does not exist. Yeah, it's not subs strings. It's subs string. Like this. Yes, exactly. You can see we get the first five characters, one to 5, 1, 2, 3, 4, 5. And if there's four, then we'll get four and a space. . Or we'll get all the four. Yeah, this is quite nice, substream, and we'll show you ~~with ~~with the slicing operator also.

You can for example, you can do like this. I will copy this one, and now we use it with the slicing. Here we have let's see. I can use, trim the word directly, trim [00:15:00] the word or slice 1, 2, 5 as sliced five cars. Copy this one, and it works equally as you can see. Code. We can also reverse characters.

Let me copy this one and I will remove these two here. Yeah, trim is good. Let's see. We can do reverse on this one. Reverse.

S reversed word. Here we reverse

copy this one.

Yeah. And you can see this is reversed. We can actually do, yeah. If you read this [00:16:00] backwards, select here's from morph. Cool. Cool, cool. , Let's try another thing. Let's try to find position of first occurrence of a sub string and return.

Zero if sub stringing is not found. .

**Kokchun Giang-3:** Let's take a look into an example here to find the first occurrence of a substring. ~~We can do select description to see that , ~~we get the description and then we do in string. ~~This ~~this stands for in string. In STR description, select. And here ~~you can ~~we want to see if the subs string select is in this string, and we find out the position of it.

~~Select position. . ~~And then when you [00:17:00] have this position. You can check if it'll give zero if it's not found and it will give ~~you can take a look at what we have now. From staging dot sgl glossary, ? Copy this one and put it here and run it here. And you can see it becomes zero.~~

~~And otherwise it becomes ~~one position that is not zero, ? Because here zero it's not found. And then here the select is in the 12 position here, or that index 12, ? That means that we can do a condition here and say that select. Position not equals zero ~~as about select. ~~That means that ~~if ~~if select exists or not.

It will give true if it exists, ? Otherwise it's false. This is a simple check to see that. ~~Yeah, ~~we have select inside of this description. ~~And ~~and now I will go into, ~~yeah, we can also, ~~I have showed concatenation before, but since I have this in my notes, I will have this here as well.

~~Fun joke. This is a concatenation, you can see here. ~~Fun joke. And you can, when you do select ~~without the, from. ~~Without the from class~~ you can ~~you can get the data directly, you can [00:18:00] select just the string and then concatenate with another string that works to get some simple answers directly.

Like you can have~~ s umca ~~string and you can see concatenated string. Fun joke. Cool. , Let's move on now to. ~~Let's see. ~~Slicing. Yes. I've shown some slicing already. We can do select~~ s scale ~~word and then ~~S scale, ~~word of zero. ~~Scale ~~word of one. ~~S scale, ~~word of minus one, for example, from staging glossary.

, Now I will just clean this up because my auto formatting doesn't work with DDB syntax, which doesn't exist in other ~~sk ~~l dialects. , Copy this one, and here, let's [00:19:00] go in. And run this one. And you can see here that, , we have some empty spaces. Here is the sq L word and ~~the m ~~we get empty space here.

~~Last word, ~~last character, empty space. And you can see first sq L zero sq word zero. We get nothing because we start counting from one. This is basically indexing, ? We have~~ s scale ~~word, the minus one. We get the last character here, which is capital E ~~here ~~and ~~M ~~for this one. ?

And we can also do, I'll copy this one and we will have S scale word. And now we'll do some slicing. Colon two for example. And we can do colon two, colon five. Copy this one. This is some slicing and as you can see here, this gives us two, one and two. Group by, we get the gr order by we get, or, and [00:20:00] then we have two to five.

It's inclusive, two, and then it's inclusive five as well. Both are inclusive. It's a little bit different from Python slicing, ? But still the syntax is quite similar. . Can we do for fun colon minus one, just to check. No, it doesn't work. ~~It's ~~but in in Python colon minus one, you will reverse the stream.

Instead, we have the reverse function here. Remember that. , We have done slicing and now it's time for doing some pattern matching. , Pattern matching is quite interesting.

Let's do pattern matching here. In pattern matching we can use the like ~~operator ~~operator in where. ~~Wear ~~class. This one the like operator used in ~~wear ~~class is used to filter roles, ? But the like operator, ~~it, ~~it can [00:21:00] be used with wild cards to search for a pattern like operator in wear class.

It will filter rows. And now we have like operator with wild cards to search for a pattern. I'll show you how that works. Let's do select from staging dot s scale glossary. And here we can use where? Clean. Let me see. ~~. Let ~~let's take an example. ~~Example, comma . Yeah.~~

Example, and then comma, clean example, which is basically trim of example. As clean example. Clean example. . And I will make it into lowercase as well. , Let's start with this one. Copy this one. Add new [00:22:00] cell, run it here. And you can see, ah, here we have example. And here is the clean example, which I removed using the trim to remove~~ left and .~~

Leading and trailing spaces. However, the middle spaces we haven't removed, now let us continue where I want to search for. . Say that I will keep this one, but now I will use this to ~~search ~~search for select.

, Here where Cleaned. A cleaned example, like select and then percentage. Look what will happen now. Now we'll find everything that will have select, and afterwards it'll match whatever character. If I paste this in, you can see, ah, [00:23:00] we get only those with starting with select ? If we want to find even those that start with other one but has select, we have a parenthesis before as well.

If I run this one, actually it didn't come up here because we didn't have it, but if it had then we will find it as well. Cool. And remember we did lower that everything becomes lowercase. Then you search for the select. If you don't do lower and you want to search, you can do eye like, which means in insensitive case you, you can still find it.

. I think it was called, I like. Let's see. For example, we can use this wild card now to represent Wild Card. ~~There, ~~there's for this one actually, we have this is the wild card like percentage, which [00:24:00] matches zero or more. ~~This matches zero or more CarX. ~~This percentage matches zero more characters, but you can have other wild cards, for example, wild card underscore.

This ~~matches ~~matches represents one character. ? I can, for example, copy this one and instead of using, I will change ~~the ~~a little bit and I will do it like this. S on the score, LECT, and I would remove the lower here as well. .

I will remove this comment here. Copy this one.

Paste it in here and you can see, ~~ah, ~~we only get those that that is. There's a character that is in here, ? S whatever character which we see, we match both [00:25:00] lowercase and uppercase. It could be, for example, a instead of e it could be whatever character, but it matches it here with this underscore.

That is quite cool. Let's go on with more pattern matching now. We will go on with pattern matching with Reg Ex. ~~Regular expression. ~~Regular expression, actually ~~make the terminal smaller. ~~To do this one ~~I will actually, yeah, ~~I will write a new ~~select. ~~Select. Lower trim of description as cleaned description from staging dot sq l glossary, where here we use rig X matches and we do clean description and we.

Match this special character here. [00:26:00] Carrot C. This will filter row starting with C, and we have made it into lowercase, ? We can copy this one, paste it in here and run it. And here everything starts with C.

And we can do it similarly. I will copy this one, and now we will start with C or S. . And I now I will have uppercase, c, s, and then I will remove this lower here

and copy this one.

Placed it here. And you can see both c and s is matched. But the uppercase ones. , Cool. And let's try to have I'll use another one now. Let's see. I will copy this [00:27:00] one.

Now I will have a, , the one before was starting with C or S. Here is we use back slash B, and what it does is that it makes it match exactly select word.

For example, it doesn't match, does not, doesn't match selects. It matched exactly the select word if we choose select. , What I mean here is that here I will do the match of select tag slash B. We have Rex matches. Clean description, select back slash B and instead of trim here, I would've lower.

. Lower description as clean description. Yeah. Yes. Copy this one. Paste it in here and run this one. [00:28:00] And you can see we find exactly select there. Select select, and all other it doesn't find . How about do we want to match a range of characters? Maybe we want to match A to F. This matches range of characters.

And if we do with carrot here, it means that it matches range of characters and it matches like starting with characters A to F. This carrot means like that. Let's copy. Let me see. Lower trim. Yeah, I'll copy this one

here. I have lower trim description as clean description. Everything is same. And here I will start with A to F. . I will [00:29:00] only have those. Copy this one. Add, sell, paste it in. And you can see here it star, it has F, C, A, C, F C, C. There's nothing more. There. There's between all of them are between A and F.

Great.

And also ~~we can do ~~we can use the knot, I will copy this here and show you that. When we do it like this, we have a carrot here. It means it matches any character, not in the range of A to F matches any character, not in range. A to F And similarly, if I have a carrot on the A, here we have matches [00:30:00] means.

It's like starting with care characters not in range A to F. If I just add a carrot here, you can see what this will mean. If I add this one, paste this one in, and you can see, ah, all of these characters here, they start. With knot A to F, ? This this means like that when you have the carrot inside of when you have this range here and you have car carrot inside of the first character there.

Yeah. There's a lot when it comes to regular expression. Let's continue. We will use reg replace. This is quite interesting. We will do select. Description, reg X not matches. Now I won't do matches. I will do [00:31:00] replace description, comma, we have one space. Plus means many spaces, one or more spaces, they shall match with only one space as cleaned.

Cleaned description, . From as staging sq l glossary. The new cell here. I paste this one in and I run it and you can see, ah, why doesn't it work here, ? That is because this will only work on the first occurrence, ? Instead we need to have the G flag to make it into global. ?

If I paste this in, ah, now you can see all the spaces in between here. Change to one space. If you also do, [00:32:00] instead of just description, we have trim of description to remove the leading and trailing. Now you can see how that looks like. Ah, better now it seems very cleaned. Super nice. Yes. Now we'll go into actually doing, making some of this cleaning and create another table inside of the refine layer. I'll show you that now.

**Kokchun Giang-4:** , now it's time to do some cleaning. We'll go into ~~clean glossary dot ~~clean glossary. Here we will do basically transformation. Let's create a new schema. First schema, if not exists. Refined. . And in this refined schema you have two different layers ~~in ~~when we're talking about data warehouse you talk about the staging layer as the first one, and then you have a refined layer or warehouse layer, and then you have.

A Mars layer, [00:33:00] which is served to the downstream, ~~the ~~consumers, . Of your data. However, now we're in the refine layer, we do some transformations based on all the exploratory data analysis and all the test cleaning we did before, ? I will make it very simple now and there's much more cleaning, but I will ~~lead, ~~leave that for the reader to do because I don't want to take away all the fun for you.

Create table if not exists, ~~find ~~dot s ~~scale ~~glossary as select. You should probably do the select first, but here I'll do upper trim of Israel word as. S word. . And the description, I will leave it as it is. We can maybe, ~~yeah, we can make a simple, we can ~~trim it. And you [00:34:00] could do the Reg X as well, ~~?~~

~~Yeah, we can do that also let's see, how did I do that? Yeah, X replace. ~~Rig X replace. Trim description, and let's see, plus this space, and then g here, and then parenthesis as description.

~~Let's see. ~~Example from. Staging s glossary. , First I will do the select statement here because I want to make sure that, let's see, can I, yes, I can do shift tab. I want to run this one to make sure that I get what I want. ~~, This one is cleaned. ~~This one is basically cleaned and this one is, it's not clean. I will leave that the example for you to clean. This I have cleaned and I'm satisfied with this cleaning. I will copy this and create [00:35:00] a table out of it. Create table, run this one. Let's see. Schema. Yeah, I need to have this schema as well. .

. Now I have created this one. Now I can do, for example, desk and you can see. Yes, I have in strings, I have this refined layer, ? From refined dot refined SQL glossary. And you can see, yes, this is the clean data. Cool.

~~ make sure that you cut out ~~make sure that you, shut down the duct DB session and then you will see there you might have a DO valve file left and you just do duct DB and norm, like the normal duct db and your let's see, cul es strings dot db, and then you do Ctrl D again, or Ctrl C if you're on windows.

This will make sure that the dot valve file disappears and your data is intact. And now make sure that you commit them. Push to your GitHub repository. That you have ~~your ~~your [00:36:00] files saved and version controlled. . I hope that you've learned a lot in this video. We have gone through strings.

We have gone through different string functions. We have. Tried to clean this very messy SQ terms that CSV. And then when we have cleaned it, we created a new schema for it, a refined layer where we put it in. And we tried out several string functions or text functions. And we also tried out slicing and indexing.

And we also tried out pattern matching and regular expression. These were a handful of different things and I hope that you have learned a lot in this video. At least I think it was really fun to record it. See you the next one. Bye. ~~I.~~

