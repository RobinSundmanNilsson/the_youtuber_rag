# SQL analytics course with DuckDB - strings concepts

[00:00:00] Hello and welcome to this lecture where we'll go into the strings concepts in ddb. ~~You have, ~~you may have heard of ~~like ~~data types like var car in databases, and ~~you may, if you are in ddb, ~~you may have seen var car, you may have seen text. And you may have seen string. These are actually interchangeable.

We'll go into these ~~in ~~concepts and then we will go into slicing and working with strings and also going into regular expressions. There's a lot of things you can do with strings ~~in, ~~in inductive B in particular. And if you are coming from Python, then you will ~~be ~~knowing that ~~the, like ~~the concepts of slicing~~ it's ~~a little bit different in inductive B since we're counting from one instead of counting from zero.

Let's move on to the slides and to the concepts.

**Kokchun Giang-1:** Here I am in the slides. Working with strings in SQL and DDB to format and clean text. What does [00:01:00] this mean? Let's find out. Text data types and their distinctions in ddb. As I said in the introduction, there are free data types for strings, but basically they're the same. We have var car.

Variable length characters. This is efficient for general text storage and it's variable length character string. It means that there is no length limit actually. ~~In other databases you ~~in other SQL dialects, you may have seen oh, var card, there's a length limit. But actually in Duct db, there is no one.

Instead here you use var, for example, VAR Car 20. It's only for documentation, it's not enforced, which means that it's not enforced. This is documenting there's 20 characters as a max, but actually you can go through that. And for, there's. [00:02:00] Aliases for var car. And these are text and string.

These are used interchangeably. You can you can create a column with text string or var car. It's the same thing in duct db. . And then we move on to using array, like slicing and indexing to get subs strings. You note that in previous lecture I've used the substring function to get the substring in for for a particular column.

You can actually use slicing similar syntax to Python if you are used to that. Here you have, for example, SQL word and you have SQL word colon two. This will give you, you slices from index one to two. Note that SQL counts from one. And here you slice this from index two to five. It gets the substream composing of characters on index two to five.

. And [00:03:00] here you can use indexing using the bracket operator. Returns empty string. If you take zero as there is no string in zero position, and here it gets the last character in the stream. , Similar to Python. This is quite nice. We can use the string functions to transform text.

For example, here you want to transform you, you use different string functions, you can use upper trim. Inside of the parentis goes first. Trim, S, k, L, word, comma. This is a space, it trims the excessive spaces and then you make upper, and then it becomes uppercase. A streamed word. Here you can get streamed word of one, and here you can get streamed word of minus one.

These are you get different characters after you have cleaned it, right? Here this is composition of [00:04:00] functions or nested functions. That is possible in S scale L and duct DB as well. And many useful text functions in the documentation. This is from the documentation directly.

Here you have several text functions which you can use. Take a look into that. I'll leave it in the description. It's good to find out what type of string functions exist. What can I use for my, for cleaning my particular strings? And then we'll have something called RegX. Regular expression for more complex pattern matching.

You may want to match a certain type of text data. For example, you use it by using RegX replace. Here you are use trim description. ~~Trim ~~trim the column. ~~And here you use here. ~~After TriMed ~~it ~~it removes the space, right? And then here I use reg Expert Place. I replace one or more spaces with one [00:05:00] space ~~and ~~and here I use similar thing, but with lower we make it into lower case as well.

, Here it replaces every matching of the pattern. This pattern, it's a space plus, right? One or more spaces and replaces it with one space. This is a way to clean data when we have several spaces and irregular number of spaces. I'll show you that in an example later on in the coding where we'll try to do this.

And then we will talk a little bit about schemas to divide into layers for structure. Here I create a schema called staging. Here is usually where you have your raw data, you put in your raw data into the staging. ~~And then afterwards. ~~The staging schema is used to represent the staging layer, a landing zone for the data.

I talk about it here because I want to introduce the concept slowly and we will go~~ we will go to ~~[00:06:00] towards more data warehouse like structure with, more layers where you have a staging layer in the beginning and then you have a refined layer and then you have something called the Mars layer.

But that will be later on when we come to data warehouse course. ~~But ~~but you should know that staging is usually used for a landing zone. Here I put in the data, and then you could have a schema that is called refine, where you have actually done some transformation here and you have.

Clean the data. One thing to note is that usually you don't use duct DB directly to transform all the data. In many cases, for example, you could connect Duct DB to DBT and use DBT to actually transform the data into the refined layer where you use the Duct DB as ~~a a. A ~~just a database. Using DBT, then it will translate the SQL into duct db.

Here you have another [00:07:00] schema that's called refined. And for that you create a table and a same table as a same name as before, but it's inside the refined layer or the refined schema. Where you have done all your cleaning here, right? Refine layer or warehouse layers

In this lecture we've gone through strings. You know that string, var, car and text is the same and you know that there is no length limit in var car. It's just for documentation and not enforced. And then we took a look into some slicing and some string functions, which you can use to transform the data and clean the data.

Talking about cleaning the data, it's good to divide it into different schemas. You have one that is staging where you have the raw data, and then in the next schema, or the next layer is the refined layer. Where you have your clean data ~~and ~~and I talked a little bit [00:08:00] about the DBT as well. Maybe in the future you won't be using DDB alone for this.

But for now, since we're learning about ddb, we use DDB to transform as well. Yes. I hope that you learned a lot in this video and see you in the next one where we'll actually go in and code out string and the string functions. That would be very interesting to use string functions to clean data.

See you there by.

