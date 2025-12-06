# Terraform setup

[00:00:00] Hello and welcome to this video where we'll finally come into IAC infrastructure S code. You are used to going into the Azure portal and clicking on different resources to add resources. Then you click on a lot of buttons to do a lot of different configurations and on and forth.

And then afterwards, you used to ~~this. De ~~delete the resources because you don't want it to cost. Can this be automated? That is the question. And if we can automate it can we then make it reusable and change it and version control it? Yes, we can. Using IAC infrastructure S code, ~~and.~~

The software, the program that we'll use for IAC is called Terraform. Terraform is used for infrastructure as code ~~for ~~it's ~~a ~~cross platform, you can use it for Azure, you can use it for GCP, you can use it for a ~~VS ~~and a lot of other places. But we will use it together with Azure in this video.

And we will, go through how to set it [00:01:00] up that you can get started with the fun world of infrastructure as code. Let's move on to my repository, because that's where we'll start and I will refer you to different links.

**Kokchun Giang-1:** here I am in My Repository. And you can see here I have a setup for Terraform, for Windows, and one for Mac. And here set up CLI for Windows and one for Mac. Okay. And in this video I will go through since I'm in the Mac, I will do the actual commands in the Mac to show you however.

It's similar in Windows. It's just a little bit older steps that you need to take. I'll show you where we go first. If we go into, we can go into the max setup up first. The max setup, you need to have home brew to install Terraform. If you don't have home Brew installed, you go into here and you'll go get into home brew.

And you just copy this one and paste it into your terminal, into your set shell. Okay? [00:02:00] And then this will install home Brew for you, and this is the package manager that you will use to install, a lot of different cool stuffs. And one of the cool stuffs is, of course, terraform. Then we go into this command here and note that, where do I get all these?

I get all these from, if you go into the windows we have we there, there is actually there, there is a page let me show you here. Install Terraform HashiCorp. You click on this one and here is basically, the information on how you install Terraform.

Okay. These are basically the steps that I've taken and I've just simplified it a little bit and add it in into my readme. ~~Okay. ~~Starting with the home Brew, we take this brew tap, HashiCorp Tap. We go into terminal and do that Brew Tap, HashiCorp Tap. Okay, done. And then run this Command Brew.

Install HashiCorp ~~tab ~~Terraform. Copy that [00:03:00] one. Go into my terminal, paste it in and it will install Terraform. It'll take a little bit of time.

Yeah, it's done good. It was quite fast. ~~Okay. ~~And the similar commands for Windows is that you need to go into this page here. There's a link. You go into this page, and here it says, this is for Mac os for Windows. What you do is that you download the binary ~~you download the, ~~either this version ~~if you are ~~or this version depending on your system.

~~You download and then ~~when you download it, you will get the ~~SIP ~~file. ~~What you do is that ~~you extract ~~it that you unzip ~~it and then ~~we go backwards. And ~~when you have unzipped it, ~~what you do is that you you put it into. ~~You go into program files ~~is a folder ~~in your C drive ~~and then you ~~create a new folder called ~~it ~~Terraform, and then you paste that file into that folder.

It's an e xe file, you paste it in and then you go into ~~your add ~~your system environment variables and you need to choose environment variable and click on path in system variables and click edit. ~~It. ~~Here you click new [00:04:00] and you add in the folder path to where your Terraform exe is. Basically you make ~~a pa you do ~~a path you connect it towards where the folder of ~~this exe, ~~this terraform file binary.

This means that you will be able to write Terraform Command in your PowerShell or Git Bash, and then when you're done with that, go into PowerShell or GI Bash and type Terraform version ~~if, ~~and then you'll get. The version of it. And also, here's a video showing these steps above. It's not made by me.

You just go in there and you can watch this video. It's very well done, I think. ~~And ~~and it shows these steps. But for me here for a Mac setup what we'll do here is that let me see. Yeah, exactly. The max set up. Yeah we do this installation and then we are finished with Terraform, what I do is I will clear the terminal and I will do Terraform version, [00:05:00] and you can see directly I have one point 13 dot three. Good. It works. And then we have the Terraform installed in our system. Okay. But that is not all because. What I said before is that we'll use Terraform together with Azure, right?

And in order to make use of Azure the Terraform needs to it depends on the Azure CLI. What you need to do is that you go into ~~here, let's see, in ~~terminal, you need to run this brew update and brew, install Azure ~~dash ~~CLI. I'll copy this one. Go back to my terminal and I'll paste it in Brew, update and then Brew.

Install Azure CLI.

~~Okay, ~~it takes a little bit of time to install the Azure CLII think. But I ~~will go into, yeah, it's done actually. ~~But I'll go into the official documentation here because this is also, it's based on this documentation from HashiCorp. And what I did is~~ he ~~simplified some steps only.

Here you can see it [00:06:00] requires this. Terraform and ~~this ~~Azure CLI to be installed. Okay, ~~in ~~to install Azure, CLI, you can see the power shell. There's a command here that you should copy for Windows and for Mac OS is this command. And then what you do afterwards, you authenticate the Azure CLI.

I'll show you how to do that. Now this is finished I can clear my terminal and I will do Azure login.

**Kokchun Giang-2:** After I've done Azure login, you will get a popup where you will log into your Azure account through the browser. ~~And do that. ~~And after that you will ~~get this you'll ~~get some instructions on which subscription to choose, and you choose the subscription that you have the ones that you want to use.

After that, you~~ co ~~make sure to copy the subscription ID because we'll use that later on. ~~I have copied my subscription id. ~~Using the subscription id, we will go into the next steps here. ~~I'll go back to my, here in my notes. And ~~you can see what we need to do is set the account subscription.

You need to do Azure account [00:07:00] set ~~subscription. ~~Subscription and here I will paste in my subscription ID and ~~here ~~it has set my subscription. ~~And then also this is very important. ~~Before you work with Terraform, you should set your environment variable in the shell or ~~go on, go into ~~go into dot set, that's HRC to set it ~~if it's ~~if you are in Mac.

~~But what you can do is that I will actually make a, or I will do it. ~~You just do basically export this variable here and you add in your subscription id, and by doing that ~~you are, ~~you'll be able to run your Terraform commands. ~~Okay. ~~And for the windows it's similar. You just copy this one and you paste that one.

This will install Azure, CLI for you. And then you can do Azure ~~login as a Z ~~login and you can do a Z account set dash, the subscription, and then you put in your subscription id. And simulative. But for Windows you do like this. ~~You put in this actually this is typo. I don't, I will remove that later.~~

There's no set as HRC in, in windows. Instead you do this one. And environment [00:08:00] arm subscription id and then ~~ne ~~put subscription id. And you do that before you work with Terraform. Okay. ~~I will do ~~I will go into ~~vicious video ~~code now and continue there.

**Kokchun Giang-3:** I'm inside of Visual Studio Code and we start with going into the extensions and you type in Terraform. What you need to do is you need to install these three, Terraform, HashiCorp Terraform, and Microsoft Terraform. This will help you in your development. It'll help you a lot with syntax highlighting, et cetera.

~~It ~~you should definitely install this ~~free. ~~And after you have installed them let's create touch.gi. In norm, you should definitely work in a ~~git ~~git repository and in a GitHub repository as well. Or it should be version controlled. And when you do that, it's important that you don't version control all the, a lot of [00:09:00] Terraform stuffs I want to put in here

**Kokchun Giang-4:** We go into Google and then just search for Terraform Gi, ignore, and you will have this GitHub here, A collection of useful git ignore templates. We go in there and you can search for Terraform and Terraform dot Git. Ignore. Go in here, and then you just copy this, everything here, and then go back to vicious digit code and paste it here.

Here you have different things for Terraform that we should not put in version control, right? And also I will put in something called ENV variables. Sh. I want to make a shell script here which I will use. Actually I won't do this because it's not the same for Windows and Mac.

Okay. ~~Yes. ~~Starting with this kit, ~~we have done this kitting, or ~~we have done this extension. I install those. Now it's time for us to [00:10:00] actually ~~create ~~create here in the setup, terraform, we can do terraform in it.

The director has no terraform configuration files. You may be. Begin working with Terraform immediately by creating Terraform configuration files initialized in Terraform. 

**Kokchun Giang-5:** This means that we are lacking a TF file, basically. ~~The touch Main tf. ~~Main tf, okay, now we have a Terraform configuration file, and now I can do Terraform in it. Then it says Terraform has been successfully initialized. And we need to actually put in some things here. Let's go back to the browser here.

And let's go back to let's see. Yeah. Here. Okay, we go, I've taken this from the official documentation, I will click on this one. And you can see here we have, build [00:11:00] infrastructure. We go down a little bit. What we did is that we have done Azure login, we're inside and. And then ~~we have done we have done, ~~we set the subscription.

And based on that one, we need to set environment variables. Actually the, we don't need to set all of this. This is suffices with arm subscription id. And here you can see this is for PowerShell. This is for Mac os. Okay. And then here, now it's a time for here you can see right the configuration.

The TF files, they're called configuration files in Terraform language. Okay, what you do is that here it just creates a folder and then CD to that folder. Fine. And then here it may not tf, they want to have this type of configuration. What does this mean? I'll copy this and I will explain a little bit, vs.

Code paste this [00:12:00] in. Here you have Terraform Block and and you have the required providers, which is Azure. Azure RM is the, here you have the source HashiCorp Azure rm. And which version it should be at least. It's larger than 3.0 0.2. And. Here is a required version of the Terraform, and then you have which provider you use.

We use the Azure rm, which is this one. This is just configuration that we can use the Azure. And here we have the what this is actually is Azure. I'll show you how to write this from scratch here. ~~When we, ~~now we want to create a resource. You can do resource and you can see there's auto complete thanks to these extensions.

We have Azure r resource group, and you can see this one, I can tab it and then the name, [00:13:00] we can call it rg. This is the name of the Terraform variable, but it's not the name of the actual resource group or actual resource. The name of the resource is name. Here I can call it my, or my tf I would just use the same my TF resource group. I use the same as the one provided in the documentation and the location. I will choose here, Sweden Central. That is it. And now when we have this, what you should do is, okay, save it first, and then you do terraform in it.

And now you can see there's this dot terraform lock dot hcl ~~l ~~and then there's this dot Terraform folder with another~~ let's say with with a ~~file here. Terraform provider ES Azure [00:14:00] rm,

**Kokchun Giang-6:** this dot Terraform folder wa it is used for local caching and operational data. Basically it's our provider plugins, modules, et cetera, is stored under this terra dot terraform folder. And this dot terraform lock dot HCL it locks the provider version for consistency and reproducibility.

That's basically what it is. And when you do init, you initialize these files based on your configuration. All of the configurations in this folder and like all the.tf files basically. And then when you have done init, you do terra form plan. To plan for what are the types that, what are the things that we want to create?

And this creates this terraform TF state, which is basically the state of our infrastructure right now. And then it, you can see here, [00:15:00] plus create, it will do plus resource, Azure RM Resource Group RG plus id. These are the new things that have been added, and what it'll do is plan one to add zero, to change, zero to destroy.

~~It, it will add a ~~it will ~~do a create here. To ~~create a resource group. It says here, Azure RM Resource Group dot RG will be created. Okay. Now we do Terraform apply to actually apply this infrastructure. We plan to see what is going to happen. Now we actually do the ~~applica ~~apply and it will prompt us what will happen now.

It says that okay, Terraform used the. Selected providers to generate the following execution plan. Resource actions are indicated with the following symbols plus create. Okay, it'll create all this or basically this resource group. Plan. Want to add zero to change, zero to destroy. Do you want to perform [00:16:00] this action?

Enter value? Yes. You need to write YES. Okay.

And when we have done this, it will create our infrastructure. ~~Oh, there's an error here. ~~You can see a resource with the id, ~~blah, blah, blah. ~~Okay. It already exists because I already created it. ~~Of course. ~~I'll just call it~~ group I would call it cool or ~~cool resource group Terraform, something like this.

~~I'll just change it. I don't remem, I don't mind the different conventions. ~~Now I'll just do terraform apply. ~~Okay.~~

Actually I can do ~~dash auto com auto auto complete, yes. Or ~~auto approve. ~~I could do that ~~I don't need to type Yes. Now. Basically, ~~okay, ~~I create this resource group.

~~Okay. ~~Yes. Let's see. Yes, you can see it's creating Azure RM resource group dot rg creating. ~~Okay. ~~Apply Complete one. Added zero changed. Zero destroyed. Okay. Let's go into Azure to look into this resource group that has been created.

**Kokchun Giang-7:** Okay, you can see here, wow, cool. Resource group dash Terraform. Yes. It [00:17:00] has been created and it is empty. Of course, I didn't add anything other inside this resource group, but it was cool that we managed to create it through Terraform. You see infrastructure as code. I write code. I can.

Commit and push this code that ~~I, ~~my infrastructure is stored in inside of this file. This infrastructure we did this and it created it in Microsoft Azure. ~~When you're done with this, ~~you want to save money. Of course, the resource script doesn't cost anything, but you may have a lot of resources that you have created in Terraform.

You just do Terraform Destroy, and I'll do that. And it will destroy ~~this ~~the resources that it has ~~been ~~created.

Yeah, it takes a little bit of time, but it shouldn't be any issues. Yes, you can see it's destroying this one Azure Resource Group, rg, refreshing state. [00:18:00] Okay. It says that it will destroy this. You can see minus here to destroy and then enter value. Yes.

And it says it's destroying this one. It's not finished yet, but when it's finished, ~~we will ~~we will just refresh~~ Azure ~~Azure Portal. Yeah. Okay. Destroyed complete ~~one. ~~Destroyed. It says, okay, here, just refresh the portal and you can see it disappeared. Awesome. Cool that this worked.

This is like a hello world of Terraform and I hope that this worked for you as well. ~~I.~~

Okay. I hope that this was fun for you as it was fun for me. ~~Of course. ~~Setting up Terraform and setting up Azure CLI and then exporting the subscription id and then we were able to create a simple configuration file, a main TF. And then we just copied in some code from the [00:19:00] documentation to basically create a resource group into Azure and then ran Terraform in it, Terraform to create all ~~the ~~the files that is needed, and then Terraform plan to see what is going to be created before you actually.

Create the infrastructure and then you do terraform apply, and then you can see the infrastructure actually is building. The thing we're building is just a resource group inside of Azure, but ~~makes ~~remember that if you are creating other resources, then ~~there, ~~there is. Cost as you would do normally ~~when you build ~~when you create resources in Azure Portal.

Yes. And then finally we destroyed our resources that were created using Terraform Destroy, ~~and ~~and then, basically, if you want to recreate those resources you just do terraform apply again, and it'll create all the infrastructure for you again. This is really fun to work with.

I hope that you will enjoy Terraform and follow along with more Terraform videos. There will be [00:20:00] many. ~~I think ~~I really like to work in Terraform and plan to create a lot of ~~in ~~infrastructure as. Code as I see this is much better than to work only ~~with ~~with the portal because you do want to version control your infrastructure.

You want to be sure what you have done. You want to be able to put it into CICD and ~~have, ~~have the infrastructure as a part ~~of ~~of your development flow. I hope that you've learned a lot, and thank you for watching this video and see you in the next one. Bye.

