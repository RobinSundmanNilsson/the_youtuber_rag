# Fastapi CRUD app

**Kokchun Giang:** [00:00:00] Hello and welcome to this video where we'll go into Fast API in Python. This is a way to create ~~an ~~A ~~PIA ~~rest API and that is what we'll be creating a ~~acru ~~application for. For a library or for books that we can serve books, data. And if you've seen some of my videos before, in a previous video, I used pedantic together with Gemini in order to generate the JSON datas in a structured way.

And then I simulated a lot of books data. Now the idea is that I will pick this JSON data and then I will continue on to work and build. An API that we can serve these books to a front end. And then since it's an API, the backend is in Python, we create the backend where we do different data processing.

We create the API, which is a layer in between our backend and. Any front end application. You could [00:01:00] use, for example you could use React, you could use Typi, you could use Plotly, you could Plotly Dash, you could use streamlet, et cetera. Or you can use yeah, whatever you want as a front end and just connect to this API.

That is the idea. But this is, a very basic one. We'll create a crud application now.

**Kokchun Giang-1:** here I am in Visual Studio Code and~~ besides, ~~besides this data here, it's empty. This is from a previous video where we used the Gemini ~~together with ap ~~together with Edan in order to create structured JSON data. And what we did was the library do Jason. Here you can see ulu Library, and here we get all the books.

And the only difference is that from that video is that I added IDs. I've added IDs for each of the books. Otherwise, it's the same data as we've gotten from that video. I will link that in the description as well. Otherwise, this is a completely empty repository or completely [00:02:00] empty folder.

And we create a terminal here. We'll create a new virtual environment, UVNV, and then Source Van Beam Activate. We activated it. We have fast API crud. This is, we will create the crud, API. That means create, read, update, and delete. In order to do that we will start with installing. UV PIP installed we will need to have let's see, what do we need?

We need to have fast API, of course, UV corn to run fast. API. And we need we can start with this one. And if we need something else, we'll install it on the way. Okay, then let's start with creating our A-P-I-A-P dot pi. And here from fast API import fast, API we will also do like this. We will create a.

Constance Stance pi, and inside of the [00:03:00] Constance Pi, I need to have the data path. From path lib, import path, remember, we're in normal Python script, and in order to define a path here, we will need to have path lib. Then data path. Data path equals to path of d the file. It's this is path to this constant PI and then dot parent.

And then you can, you are in fast, API crud. You can see all of these folders. And I will go into the data folder. Then this data path exists here we can import it in in elsewhere. And also let's see. We will need to have I will need to have another script here. I'll create another script called DataCore Processing pi.

The reason for doing this is that I don't want to put everything into my API module. I want to separate it and and make it more [00:04:00] scalable. We will start with reading in the data. Import j we will take from Constance import data path and then we will do def read Jason, and here I'll place in the fine name and with open data path slash fine name.

And I want to ~~read it. ~~Read it. We take R as file, and then data equal to js dot load file. And then we return data basically like that. We can try this out. If d the name equals to D the main, then we'll do read Jason. And what do we want to place in? What is this called? It's called Library [00:05:00] Jason.

Library J data goes to this, and then I will print the data and I'll run this one. Okay. You can see we have the data here. If we want to print a little bit nicely, we can do from P print, import P print. And then instead of print, I will do p print instead. Clear the terminal and run it again. Okay.

Yeah, it still looked, oh. I had print here. P print. Oh yeah. Now it looks better. You can see books. We have. Name also. But in inside of books we have a list of dictionaries. It's a list of dictionaries. Since it's a list of dictionary, we can do it like this. We can create a class called book.

This is our model, our book model. And what we need is from P den [00:06:00] import base model. Base model, comma field. And this book, I will call base model. It's it's inheriting base model. It means that this one will become a a edan model, but it's still a normal class. It's a class, but it's a Edan model as well.

Id in. You can see inside of books here, it's for each book you can see we have id, we have author, title, and year. Okay. And then we can have title, it's a string. We can have author. It's a string as well. And we have a year. It's an int and this is a field. I want it to be greater than equals to greater than 1000 and less than equals to this year.

2025. And to get this year, we can do current year plus one. And [00:07:00] to get the current year let me just go into Constance and we can do from daytime import daytime, and we can do. Current year equals to date time now year. It means that data processing, we can import that from stance import current year.

And what this means is that~~ we have a. We, ~~we want it to be larger than, greater than 1000. And we want it to be less than 2026, since it's 2025 now. This current year will become 2025 and plus one, we'll have 2026. Okay? This is our gigantic model and ~~for ~~for our book. ~~And then we can do, also, we can do.~~

Class library. It's a base model and the name is String and [00:08:00] Books is list of book, we have this book here that is a model. Library is having the name and it has several books. If you're not familiar with pedantic, make sure to take a look into my pedantic videos ~~before ~~because~~ they will, ~~they are the foundation for fast API fast.

API uses pedantic a lot for validating the data. Then we have. Library here, and we can also do like this. We can do deaf library data or file name and then Jason. Jason. Data equals to read Jason. I use the read Jason by placing the fine name and then return library. ~~Model validate ~~model, validate JSON data.

Okay, this means that [00:09:00] the library will ~~we, ~~we can do library data. It will read this ~~Jason ~~file. And get this JSON data. And then we'll do library do model validate. This means that instead of doing ~~re ~~json like this, we can try this out by doing library equals to library data of let's see, what's it called?

Library J.

And we can take a look into library. What does that look like? If I run this one, you can see it's a library object where we have the name Ulu Library, we have books. And we have all the books here. And each of the books contain a book object or a book instance where we have ID title, we have author and year.

These [00:10:00] are all ~~edan ~~models. All the book are the models. Library is a pedantic model. That means that we have already validated the data that it comes into the ~~right ~~right fields. We have in string and we have interior with the correct fields here. If you, for example, take a look, for example, if we take 19 ~~hundreds ~~here, we'll have a fail here because you can see there's a lot of books here.

Books one year. You can see input should be greater than 1900 books. Five point year input should be greater than 1900, et cetera. And here you can see it's 18 13, 18 51. You can see there is, it's validating the data, but ~~the ~~I want to make it simple. I just take 1000 here. If I run this one, it should be fine because all have years greater than 1000.

Okay, for this I'm quite satisfied with the data processing. Let's move on to our API. Inside of the API dot [00:11:00] pi, you start with creating an app equals to fast API. ~~This ~~now we have a fast API application, and then. What we should do also is that we'll read in the data. I can do it before actually library equals to, from data processing, import library data, and we can do library data here.

~~Library J. ~~Then we have the library, and then we can get the books. And if you see library to get the books, we do library, do books, library dot books. And basically now you can p print the, ~~let's see, ~~we need to import P print from P print, import, P print. And we can do p print of books. If I just run this one, you can see here are all the books and it's a list of the books.

~~Good. ~~And we can now we have our Fast API application and we can create our [00:12:00] first endpoint. At app get. If I go into my URL and I do ~~slash. My first endpoint ~~slash books, I should read all the books, this is a decorator, it'll decorate a function that makes it into ~~a a end ~~an endpoint with the GET request.

Get is~~ sim, it's ~~the read operation in crud. Create, read, update, delete. ~~We'll create, ~~this is a get request, then you do async dev and async is to make the the function asynchronous that it doesn't get stuck when when running this one. But fast. API actually makes all the endpoints into async by default, but I type it out to make it extra clear.

Async dev read books. Return books basically. Let's try out this endpoint. To try it out, you do like this. You have activated your virtual environment. You have installed UV, corn and [00:13:00] fast API. Let's do UV corn api, pi API Colon app. ~~API is this API. ~~You don't~~ in ~~include the dot pi and then you take what the fast API application's name, which is app, and then that dash reload.

You don't need to run it each time you change the code. I run this one and what we get is a server that runs on ~~8,000, ~~port 8,000. ~~I run this one. ~~I will have this side by side. I will make this a little bit bigger and this a little bit smaller like this.

Actually, I will have this run out like this. Okay? Right now it says detail not found, but if you do slash books, you can see, ah, these are our books. Wow. Cool. Okay, we can click indented write out the indented code.

**Kokchun Giang-2:** Yes. Much cooler. Now it's indented. You can read the data very clearly. Okay, what is the point with [00:14:00] this? When you have it you serve it in API like this. You have a URL, and you can use this URL to do, get request. For example~~ if you ~~if you are in a Jupyter Notebook and you do request on this data.

~~You will get this data back. You do request on this URL, ~~you will get this data back and that means that you can serve this to ~~a ~~a frontend application without problems. Let's show you. Another very neat thing is in fast API, there is something called Swagger ui. You go into slash docs and you come into this documentation page here in Fast, API, and you can see there's this get request here.

You can click here and you can try it out and you can execute it. And what you can see here is that to do curl. You do type like this and curl is basically what you type in your terminal. If I create a new terminal here and I paste this in this curl, and you can see I get the data. Cool. [00:15:00] And you can, this is the request, URL.

As you saw before, we have this URL and then stash books. Then ~~we get into this ~~we get the data we can copy this one and you can see we open here. And you can see we get the data good. And then the response, you can get it here. When you have Swagger ui, you can directly interact with your API through this user interface.

And this is really neat. You don't need to have a third party tool such as Postman or some other ~~to ~~tools. You can do it directly here. Okay? Let us continue building on our API. We'll create something called a path parameter. And the path parameter, you should place it before your query parameters.

I'll show you what ~~per ~~query parameters is also. App get slash books. We still go into [00:16:00] slash books and then we go into title. And we go into, we type in the title. This is how you do path Parameter. You have curly braces of ~~your your what you want to ~~what the variable should be, what the string should be after the slash title.

This one will go into Async f Read book by title. And then you will take the title, it's a string and we should return book for book in books. If book do, title do case fold. Case fold, makes it lowercase equals two. Title case fold. It's lowercase.

Let me make it a little bit bigger. What it does is that we have book ~~four ~~[00:17:00] book in books. We get all the books by this title. Where the book, ~~the ~~title equals to. Title case fold. It means that we can type in books with capital letters and it doesn't matter, it will be translated.

Let's restart this one. You can see, oh, we get another get request there. Try this out and you can see in Swagger ui you can just type in the title. What title do we have? The Great Gatsby. The great Gaby. Let me just type it like this and show that it works. Anyway, execute, and you can see we get back this de Great Gatsby f Scott Fitzgerald.

Okay, here you can see this is the URL there. And then we have percentage, 20, great percentage, 20 Gatsby. Percentage, 20 means [00:18:00] space. This is the URL. If you copy this URL and you go into here, you can see we get this data back. It works. Okay, let's continue. This was a path parameter. You could do ~~per ~~parameters or IDs also if you want.

Now let's do a ~~query parameter. ~~Query parameter. It'll, it means that you have a question mark after your endpoint. Start on this, ~~or year ~~I will use this one equals 1950, for example, if you do this. Then this variable start year equals 1950 will be used in our endpoint. At app get slash books slash this one here, we'll go into the books and put in our question mark.

Let's [00:19:00] see, async dev filter books ~~and. ~~And what I want to do is start here. This is an int equals to let me import. ~~Query. Query. It's for query, pars ~~query. For default value. This is similar to using the field~~ in in, ~~in pedantic. Query, I want it to be by default, 1950 greater equals to say 1500 less than equals to Let's take current year.

From Constance import current year, this equals current year plus one.

And then I will also have a description,

~~the script, ~~the description equals [00:20:00] to filters, books that are newer than this year. ~~Okay. ~~Okay. Let's see. This is my ~~deaf ~~and I will do filtered books equals to book for book in books. If start year is less than book ~~thought ~~year. Basically like this. Okay, I filter these books and what we do is that we want to return filtered books.

Okay? Save this one. Go into swagger, ui, update it. And you can see here we have filter books. I test this out. Okay, try 1950, execute. And you saw that. Okay. We don't get [00:21:00] any books back. Let's see why. Book for book in books. If start year is less than book that year book that year and start year is an int.

Oh, I haven't, oh. I have run it. It's, it is run. Okay. I saved this one.

Try it out. Filter books that are new than this year. 1950. Execute. I get back nothing. Let me see why. Less than greater than 1500. Less than. 2051. 2026. Filter. This book? Yeah. Oh, okay. Filter book. Ah, filter books. Yeah. I return this function. Of course it doesn't give anything. [00:22:00] Okay. Execute this one and it works.

And you can see all of them are newer than 1950. If I take, for example, 90 55 here. We will not get a lot of the rings. If I run this one, we'll only get these two. Okay, great. It works. Now we filter by this, but if you want, you can also, we can see this is the URL, you have slash books, and then we have question marks.

Start year equals 90 55. If we want, we can put in more parameters to to query parameters we can have author also if we want. Author is a string. It's a query. None you don't play. It makes it optional. And then we can have a description authors first name and last name. Filter by author's, first name and last name.[00:23:00] 

Okay, this is the description and we have author and then we can filter the book. If author, it means it's not none, then filtered books equals to this list. Book for book in filtered books. If author.case fold. Equals to book author case fold. Then we return these books here, save this one or update this one.

You can see now author comes in as well. Try it out and you can see if we just take 1950 and execute it, it works. ~~Because we haven't ~~because this is it has none as the fault, it's optional. But if I put in an author, for example, Douglas Adams. Douglas Adams [00:24:00] execute it, you can see we get only this one back.

Good. This one works. Now we only use the get requests. We are still in read. If we want to create a crud operations, we should do create as well. To do create, we do app post. What you want to do is ~~you want to put in a, ~~you want to post a new book. Books slash create book. And here, async Deaf, great book request.

You want to put in a new book as a J object, and this should be book.

Okay, and ~~we don't have books, ~~we need to import this one import book. This is the Edan model. It means that it has to conform to the, to, to the fields in the Edan model in order for it to create a new book and then [00:25:00] to create a new book. It's as simple as this new book equals to book model.

A ~~book is a, remember the ~~book is a pedantic model, we can ~~do model ~~validate our book request. Just to make sure we turn it into ~~a. ~~A book object. And then what we do is books that append new book. And remember that we append the book. We have our books ~~contains a lot of they ~~contains book objects that are pedantic models.

However, when we serve it in the API remember that it gives us ~~chase in ~~data. And what it does is that fast API automatically. ~~It ~~converts it ~~into ~~into JSON data, and that is very good. We have return new book because this is good to do. ~~When you ~~when you basically, you have appended it into your books.

That when you do [00:26:00] get request, you will see the new book as well. However when you do post you want to return a new book to show you that it's the new book that is placed in. Let's update this one, and now you can see there's a post. Go into here. We try this out and these are example values you can see.

Id zero now. We haven't done ~~any ~~any validation on the id. You could do that. ~~14, ~~for example, title is a string. ~~And what is this? ~~This example is based on our P model. You can see our book model here. It's an int that's why they have a value here, a string. It's also an author.

It's also string. You got the value as strings here and int you get this field. Here. ~~Let us make this let us just try or ~~let us make this more helpful. In the documentation. To do that, we go into our data processing and into our book model, [00:27:00] and we'll do something called model config.

It's a dictionary, Jason Schema Extra. This is another dictionary. And here we have an example and. Which is non dictionary. And now the example, let's take ID colon 11. We have a title right then with a engineer,

author co yang here. 2025. Save this one

and let us try it out now. If I update this one, you can see, ah, this is the example values. Good. Try it out. And you can see this example values. Yeah, [00:28:00] we can use this directly. Execute, and you can see that. The response body, it gives back this one. This is the response after we have posted it. Okay.

But let's see if this book is there. Read books, try it out, execute. You can see down here. Ah, 11. Learn with Engineer K Young. Okay, cool. Our post request works. And also you can try it out here if you want to read all books.

You can see that. Yeah, it's here. Now we have posted the data and let us continue now. We have. Create is post in in fast, API and we can do at app update. This is the update and we can slash books [00:29:00] slash up update book, async dev update book. Updated book. It's a book object, a book model for I book in enumerate books.

This means that we'll get indexes for each of the books. And then if book ID equals to Id.

Let's see if book ID equals ~~updated book id, ~~updated book id. If this is the case, then books of I equals the updated book. ~~Okay, it means that. We will update ~~we will create~~ when we update we put in ~~a JSON object and the JS ~~O ~~will [00:30:00] take a look into the id. If it's the same ID as another book that exists, then we'll update that book.

Otherwise, we won't do anything. After that, we return our updated book. ~~Okay. ~~Fast. API don't let's see. Oh it's not called update. It's called put in fast. API ~~it's called Put. ~~Okay, let us try to change ID 10 here. In my, put, try it out. Let's change Id 10 to. Okay, cool Book. And I've written it, execute it and it returns back this updated book.

Let's look into here. This is my slash books here. If I run this one, you can see, ah, I did 10. [00:31:00] Cool book. And why did my 11 disappear? Yeah, because I have reloaded it. Then the data is not stored in the j per se, we haven't done logic for that in, instead it's just for this run here.

If you want it to be stored, you should connect it to a database, but I won't show it here. We have done update book. What is left is that we need to delete book. Then we have our cred application. App delete slash books slash delete book slash id. This is a what's called, it's a, puff parameter, async deaf delete book, and it should take in the ID [00:32:00] as an ink and for I comma book in enumerate books. If book id. Equals id. Then we delete this book, books of I Break. Okay, let's try this out. ~~Update this one. ~~Delete id. Let's remove one. Execute.

If we go back here and take a look into all the books. You can see ID one has disappeared. Let's remove ID five as well. Five execute. And now we need to rerun this one. Execute. And you can see five has disappeared. Cool. This is our fast API application

**Kokchun Giang-3:** We have created our [00:33:00] first CRUD application ~~using ~~using Fast API. ~~And ~~here we have worked with ~~p we have worked with Jason ~~Data and we have created our. ~~Book API ~~books API, that you can read from the books get requests, you can do posts, you can do update, you can do update by put, and you can do deletes.

This is ~~really ~~really interesting ~~and ~~and what you can do later with this is that you can connect this into a frontend application. ~~And ~~and then you can, mix and match and use whatever ~~frontend application you want. ~~Frontend framework. You want ~~to do this that ~~if you want to mix Python with JavaScript, no problems.

If you want to mix Python with something else, that's no problem either. You can use Python all the way. ~~Also, ~~that's also possible. There's a lot of opportunities when you work with APIs. It's a layer in between now and this is really interesting and I hope. ~~Hope that ~~you have learned a lot from this video ~~and ~~thank you.

See you in the next one. Bye.

