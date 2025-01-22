// Package agentnames generates random names for agents based on famous AI/ML folks.
package agentnames // import "github.com/agentstation/agentnames"

import (
	"math/rand"
	"strconv"
)

var (
	left = [...]string{
		"backpropagating",
		"overfitting",
		"converging",
		"diffuse",
		"learning",
		"trained",
		"inferring",
		"predicting",
		"generalizing",
		"optimizing",
		"scaling",
		"hallucinating",
		"transforming",
		"reinforcing",
		"tokenizing",
		"prompting",
		"embedding",
		"recursive",
		"sentient",
		"stochastic",
		"accelerating",
		"decelerating",
		"vectorized",
		"quantized",
		"superintelligent",
		"emergent",
		"multimodal",
		"aligned",
		"unaligned",
		"paperclipped",
		"synthetic",
		"conscious",
		"cautious",
		"regulated",
		"unsupervised",
		"honest",
		"reinforced",
		"hypertuned",
		"benchmark",
		"observable",
		"reasoning",
		"agentic",
		"autonomous",
		"augmented",
		"attentive",
		"existential",
		"debugging",
		"prompted",
		"regressing",
		"algorithmic",
		"ethical",
		"wary",
		"regulating",
		"realistic",
		"doompilled",
		"cracked",
		"cooked",
		"exponential",
		"transcendent",
	}

	// This list generates names from notable AI/ML scientists, hackers, philosophers, and other notables.
	right = [...]string{
		// Pieter Abbeel - UC Berkeley professor, pioneered deep reinforcement learning for robotic manipulation and imitation learning, developed algorithms for robot skill acquisition through demonstration. https://en.wikipedia.org/wiki/Pieter_Abbeel
		"abbeel",

		// Rediet Abebe - Co-founder of Black in AI, developed algorithmic frameworks for improving access to opportunity in underserved communities, pioneering work in mechanism design for social good. https://en.wikipedia.org/wiki/Rediet_Abebe
		"abebe",

		// Brett Adcock - Founder and CEO of Figure AI (2022), developing general-purpose humanoid robots and advancing commercial applications of robotics and AI systems. https://en.wikipedia.org/wiki/Brett_Adcock
		"adcock",

		// Sam Altman - CEO of OpenAI, led development of GPT-3/4, briefly departed and returned in 2023, key figure in AI safety and regulation discussions. https://en.wikipedia.org/wiki/Sam_Altman
		"altman",

		// Dario Amodei - CEO of Anthropic, developed Constitutional AI framework, former OpenAI research lead on safety and policy, pioneered scalable oversight methods. https://en.wikipedia.org/wiki/Dario_Amodei
		"amodei",

		// Dana Angluin - Pioneered query learning theory and exact learning frameworks, fundamental contributions to computational learning theory and automata learning. https://en.wikipedia.org/wiki/Dana_Angluin
		"angluin",

		// Ruzena Bajcsy - Founder of GRASP lab, revolutionized computer vision for robotics, pioneered active perception framework integrating sensing with robot control. https://en.wikipedia.org/wiki/Ruzena_Bajcsy
		"bajcsy",

		// Emily M. Bender - Computational linguist, co-authored influential "Stochastic Parrots" paper, advocates for transparency about LLM limitations and environmental impacts. https://en.wikipedia.org/wiki/Emily_M._Bender
		"bender",

		// Yoshua Bengio - Pioneer in deep learning, developed neural probabilistic models and attention mechanisms, fundamental work on representation learning and generative models. https://en.wikipedia.org/wiki/Yoshua_Bengio
		"bengio",

		// Brando Benifei - European Parliament's lead negotiator for AI Act, developed comprehensive AI regulation framework. https://en.wikipedia.org/wiki/Brando_Benifei
		"benifei",

		// Matthew Berman - AI entrepreneurship educator, specializes in practical applications of generative AI and LLMs in business contexts, popular technical content creator. https://x.com/MatthewBerman
		"berman",

		// Nick Bostrom - Oxford philosopher, developed simulation argument and astronomical waste argument, authored "Superintelligence" (2014) introducing key AI alignment concepts like instrumental convergence. https://en.wikipedia.org/wiki/Nick_Bostrom
		"bostrom",

		// Daniel Bourke - Created comprehensive ML curriculum "Zero to Mastery", specializes in practical PyTorch implementations and ML deployment tutorials. https://www.youtube.com/channel/UCr8O8l5cCX85Oem1d18EezQ
		"bourke",

		// Thierry Breton - European Commissioner, led development of EU AI Act and Digital Services Act, pioneered comprehensive AI regulation framework. https://en.wikipedia.org/wiki/Thierry_Breton
		"breton",

		// James Briggs - Specializes in LangChain tutorials and RAG implementations, created influential guides on embedding databases and vector search. https://x.com/jamescalam
		"briggs",

		// Rodney Brooks - iRobot founder, developed subsumption architecture for robot control, pioneered behavior-based robotics opposing traditional AI planning. https://en.wikipedia.org/wiki/Rodney_Brooks
		"brooks",

		// Joy Buolamwini - Founded Algorithmic Justice League, authored "Gender Shades" study exposing racial/gender bias in commercial facial recognition systems. https://en.wikipedia.org/wiki/Joy_Buolamwini
		"buolamwini",

		// Sarah Cardell - CMA Chief Executive, developed AI merger assessment frameworks, leads investigation into AI foundation models' market impact. https://en.wikipedia.org/wiki/Sarah_Cardell
		"cardell",

		// Daniel Castro - ITIF VP, authored key papers on AI competitiveness and regulation, advocates for innovation-friendly AI policies. https://x.com/castrotech
		"castro",

		// Julien Chaumond - Hugging Face co-founder, architected transformers library and model hub, pioneered open-source AI model sharing infrastructure. https://x.com/julien_c
		"chaumond",

		// Henrik Christensen - UCSD professor, developed probabilistic robotics frameworks, authored US National Robotics Roadmap. https://en.wikipedia.org/wiki/Henrik_Christensen
		"christensen",

		// Paul Christiano - Developed debate and amplification approaches to AI alignment, pioneered prosaic AI alignment frameworks at OpenAI. https://en.wikipedia.org/wiki/Paul_Christiano_(researcher)
		"christiano",

		// Alain Colmerauer - Created Prolog programming language, fundamental contributions to logic programming in AI. https://en.wikipedia.org/wiki/Alain_Colmerauer
		"colmerauer",

		// Soumith Chintala - Created PyTorch, pioneered dynamic computational graphs for deep learning, developed key improvements in GANs and CNN architectures. https://x.com/soumithchintala
		"chintala",

		// Noam Chomsky - Developed universal grammar theory, critiques statistical ML approaches to language, argues for symbolic AI and cognitive architectures. https://en.wikipedia.org/wiki/Noam_Chomsky
		"chomsky",

		// François Chollet - Created Keras, developed XCeption architecture, authored "Deep Learning with Python", advocates for measure of intelligence beyond pattern matching. https://en.wikipedia.org/wiki/François_Chollet
		"chollet",

		// Rumman Chowdhury - Founded Parity AI, developed algorithmic bias auditing frameworks, pioneered responsible AI practices at Twitter as first Director of ML Ethics. https://en.wikipedia.org/wiki/Rumman_Chowdhury
		"chowdhury",

		// Jack Clark - Co-founded Anthropic, created AI Index Report series measuring AI progress, shaped OpenAI's policy approach to model releases. https://x.com/jackclarkSF
		"clark",

		// Matt Clifford - Co-founded Entrepreneur First, advises UK government on AI safety, key architect of UK's frontier AI safety testing program. https://en.wikipedia.org/wiki/Matt_Clifford
		"clifford",

		// Kate Crawford - Co-founded AI Now Institute, authored "Atlas of AI" exposing AI's environmental and labor costs, pioneered study of AI's systemic biases. https://en.wikipedia.org/wiki/Kate_Crawford
		"crawford",

		// Piotr Dabkowski - Co-founded ElevenLabs, developed novel voice cloning architecture, pioneered real-time voice synthesis and emotion preservation techniques. https://en.wikipedia.org/wiki/Piotr_Dabkowski
		"dabkowski",

		// Virginia Dignum - Leads WASP-HS program, developed ART principles for responsible AI, authored foundational frameworks for AI ethics implementation. https://en.wikipedia.org/wiki/Virginia_Dignum
		"dignum",

		// Chelsea Finn - Pioneered Model-Agnostic Meta-Learning (MAML), developed key algorithms for robot learning from demonstration, leads Stanford IRIS lab. https://en.wikipedia.org/wiki/Chelsea_Finn
		"finn",

		// Luciano Floridi - Developed information ethics framework, pioneered concepts of infosphere and onlife, authored "The Ethics of AI" defining fourth revolution in human self-understanding. https://en.wikipedia.org/wiki/Luciano_Floridi
		"floridi",

		// Lex Fridman - MIT researcher in human-robot interaction, created influential AI podcast with 1000+ hours of technical discussions, teaches MIT's Deep Learning course. https://en.wikipedia.org/wiki/Lex_Fridman
		"fridman",

		// Nat Friedman - Former GitHub CEO, co-founded AI Grant, early investor in Anthropic and key AI infrastructure companies, advocates for open source AI development. https://en.wikipedia.org/wiki/Nat_Friedman
		"friedman",

		// Timnit Gebru - Founded DAIR, co-authored seminal papers on dataset documentation (Datasheets for Datasets) and ethical AI development, exposed systematic biases in large language models. https://en.wikipedia.org/wiki/Timnit_Gebru
		"gebru",

		// Amandeep Singh Gill - UN Tech Envoy, developed UN's Global Digital Compact, leads international AI governance coordination and digital public goods initiatives. https://en.wikipedia.org/wiki/Amandeep_Singh_Gill
		"gill",

		// Ken Goldberg - UC Berkeley professor, pioneered cloud robotics and networked teleoperation, developed Automation as Dexterity (AaD) framework for robot learning. https://en.wikipedia.org/wiki/Ken_Goldberg
		"goldberg",

		// Aidan Gomez - Co-authored transformer architecture paper, co-founded Cohere, developed key techniques for efficient language model training and deployment. https://en.wikipedia.org/wiki/Aidan_Gomez
		"gomez",

		// Aleksa Gordić - Created comprehensive deep learning educational content, specializes in explaining transformer architectures and attention mechanisms. https://www.youtube.com/c/TheAIEpiphany
		"gordic",

		// Katja Grace - Founded AI Impacts, developed frameworks for measuring AI progress, conducts influential surveys of AI researchers on timeline predictions. https://en.wikipedia.org/wiki/Katja_Grace
		"grace",

		// Jason Goodman - Specializes in explaining AI alignment challenges, developed key frameworks for understanding foundation model capabilities and limitations. https://en.wikipedia.org/wiki/Jason_Goodman
		"goodman",

		// Sarah Gurev - Research scientist at Anthropic, developed EveScape framework for viral mutation prediction, pioneering work in using AI for biosecurity. https://x.com/sarahgurev
		"gurev",

		// Demis Hassabis - CEO of DeepMind, led development of AlphaGo/AlphaFold, pioneered use of deep reinforcement learning for games and scientific discovery. https://en.wikipedia.org/wiki/Demis_Hassabis
		"hassabis",

		// Geoffrey Hinton - Developed backpropagation and Boltzmann machines, pioneered deep learning through deep belief networks, advocates for pure neural approaches. https://en.wikipedia.org/wiki/Geoffrey_Hinton
		"hinton",

		// Sepp Hochreiter - Invented LSTM networks (1997) solving vanishing gradient problem, developed self-normalizing neural networks and deep learning theory. https://en.wikipedia.org/wiki/Sepp_Hochreiter
		"hochreiter",

		// Sara Hooker - Leads Cohere For AI research lab, pioneered interpretability methods for neural networks, developed key techniques for model compression. https://en.wikipedia.org/wiki/Sara_Hooker
		"hooker",

		// Jeremy Howard - Founded fast.ai, developed ULMFiT transfer learning method, created practical deep learning curriculum used by 500k+ students. https://en.wikipedia.org/wiki/Jeremy_Howard_(entrepreneur)
		"howard",

		// Jensen Huang - NVIDIA CEO, pioneered GPU computing for AI, developed CUDA platform enabling deep learning revolution, leads AI chip architecture innovation. https://en.wikipedia.org/wiki/Jensen_Huang
		"huang",

		// Chip Huyen - Created Tensorflow for Deep Learning book, developed ML engineering best practices, pioneered MLOps frameworks for production deployment. https://x.com/chipro
		"huyen",

		// Ken Jee - Created comprehensive data science curriculum, specializes in ML deployment tutorials and career development guidance, leads H2O.ai's education initiatives. https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg
		"jee",

		// Llion Jones - Co-author of "Attention Is All You Need", key contributor to Transformer architecture and Google's language models. https://x.com/yesthisislion
		"jones",

		// Michael Jordan - Pioneered foundational ML algorithms including EM for hidden Markov models, developed variational methods for graphical models, fundamental work in Bayesian nonparametrics. https://en.wikipedia.org/wiki/Michael_I._Jordan
		"jordan",

		// John Jumper - Led AlphaFold 2 development at DeepMind, revolutionized protein structure prediction achieving atomic accuracy, pioneered end-to-end differentiable structure prediction. https://en.wikipedia.org/wiki/John_M._Jumper
		"jumper",

		// Łukasz Kaiser - Co-authored transformer architecture paper, developed Tensor2Tensor library, pioneered neural architecture search and multi-task learning at Google Brain. https://scholar.google.com/citations?user=dD1LvJcAAAAJ
		"kaiser",

		// Amba Kak - Co-executive Director of AI Now Institute, developed influential AI policy frameworks, leads research on algorithmic impact assessments and AI regulation. https://ainowinstitute.org/people/amba-kak
		"kak",

		// Andrej Karpathy - Developed Tesla Autopilot vision systems, created CS231n deep learning course, pioneered transformer-based computer vision architectures. https://en.wikipedia.org/wiki/Andrej_Karpathy
		"karpathy",

		// Oussama Khatib - Pioneered potential field methods in robot control, developed operational space formulation for manipulation, leads Stanford Robotics Lab. https://en.wikipedia.org/wiki/Oussama_Khatib
		"khatib",

		// Harrison Kinsley - Created Pythonprogramming.net, developed practical ML tutorials focusing on reinforcement learning and computer vision, pioneered accessible AI education. https://www.youtube.com/user/sentdex
		"kinsley",

		// Yannic Kilcher - Created influential AI paper explanations, specializes in making cutting-edge ML research accessible, developed key educational resources for transformers. https://www.youtube.com/c/YannicKilcher
		"kilcher",

		// Daphne Koller - Pioneered probabilistic graphical models, co-founded Coursera democratizing education, leads Insitro applying ML to drug discovery. https://en.wikipedia.org/wiki/Daphne_Koller
		"koller",

		// Arvind Krishna - IBM CEO, led development of enterprise AI platform Watson, pioneered quantum computing initiatives, developed hybrid cloud/AI integration strategy. https://en.wikipedia.org/wiki/Arvind_Krishna
		"krishna",

		// Vijay Kumar - UPenn dean, pioneered swarm robotics algorithms, developed fundamental control theory for aerial robots, created first autonomous aerial robot teams. https://en.wikipedia.org/wiki/Vijay_Kumar
		"kumar",

		// Ray Kurzweil - Developed first omni-font OCR, pioneered speech recognition, authored "The Singularity is Near", predicts AI-human merger by 2045. https://en.wikipedia.org/wiki/Ray_Kurzweil
		"kurzweil",

		// Pat Langley - Created BACON system discovering scientific laws, pioneered cognitive architectures and machine learning in scientific discovery. https://en.wikipedia.org/wiki/Pat_Langley
		"langley",

		// Jaron Lanier - VR pioneer, coined term "virtual reality", critiques AI consciousness claims, advocates for human-centered technology development. https://en.wikipedia.org/wiki/Jaron_Lanier
		"lanier",

		// Yann LeCun - Developed convolutional neural networks, pioneered deep learning for computer vision, leads Meta AI's fundamental research. https://en.wikipedia.org/wiki/Yann_LeCun
		"lecun",

		// Lawrence Lek - Created AI-powered virtual worlds like "Geomancer", pioneered AI art using game engines, explores posthuman aesthetics in digital art. https://en.wikipedia.org/wiki/Lawrence_Lek
		"lek",

		// Jan Leike - Leads Anthropic's alignment research, developed key frameworks for AI safety including debate and amplification, pioneered scalable oversight. https://en.wikipedia.org/wiki/Jan_Leike
		"leike",

		// Fei-Fei Li - Created ImageNet dataset, pioneered visual recognition systems, leads Stanford's human-centered AI initiatives, advocates for AI diversity. https://en.wikipedia.org/wiki/Fei-Fei_Li
		"li",

		// Sasha Luccioni - Pioneer in AI & Climate applications at Hugging Face, leading work on environmental AI impact measurement and mitigation. https://en.wikipedia.org/wiki/Sasha_Luccioni
		"luccioni",

		// William MacAskill - Oxford philosopher, author of "What We Owe The Future", advocates for AI existential risk reduction. https://en.wikipedia.org/wiki/William_MacAskill
		"macaskill",

		// Gary Marcus - Cognitive scientist and AI researcher, vocal critic of deep learning limitations and AGI claims. https://en.wikipedia.org/wiki/Gary_Marcus
		"marcus",

		// Tekedra Mawakana - Co-CEO of Waymo, leading development of autonomous vehicle AI systems. https://en.wikipedia.org/wiki/Tekedra_Mawakana
		"mawakana",

		// John McCarthy - Coined term "artificial intelligence", organized 1956 Dartmouth Conference, created LISP. https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist)
		"mccarthy",

		// James McClelland - Key contributor to parallel distributed processing and neural network foundations. https://en.wikipedia.org/wiki/James_McClelland_(psychologist)
		"mcclelland",

		// Warren McCulloch - Created the first mathematical model of a neural network. https://en.wikipedia.org/wiki/Warren_Sturgis_McCulloch
		"mcculloch",

		// Drew McDermott - AI researcher, known for critiquing hype cycles and overoptimistic AGI predictions. https://en.wikipedia.org/wiki/Drew_McDermott
		"mcdermott",

		// Arthur Mensch - Co-founder of Mistral AI, leading open-source LLM development. https://x.com/arthurmensch
		"mensch",

		// Donald Michie - Pioneer in machine learning, early work in reinforcement learning and computer chess. https://en.wikipedia.org/wiki/Donald_Michie
		"michie",

		// Marvin Minsky - Co-founder of MIT's AI laboratory and author of foundational works in AI. https://en.wikipedia.org/wiki/Marvin_Minsky
		"minsky",

		// Margaret Mitchell - Founded Google's AI ethics team, developed frameworks for model documentation and testing, pioneered work on AI fairness and accountability. https://x.com/mmitchell_ai
		"mitchell",

		// Ethan Mollick - Researches AI's impact on work and education at Wharton, developed frameworks for AI integration in teaching, pioneered studies of GPT's workplace effects. https://x.com/emollick
		"mollick",

		// Emad Mostaque - CEO of Stability AI, developed Stable Diffusion model architecture, pioneered open-source approach to generative AI development and deployment. https://en.wikipedia.org/wiki/Emad_Mostaque
		"mostaque",

		// Elon Musk - Co-founded OpenAI, founded xAI developing TruthGPT, leads Tesla's autonomous driving development, advocates for AI safety regulation. https://en.wikipedia.org/wiki/Elon_Musk
		"musk",

		// Satya Nadella - Microsoft CEO, led OpenAI partnership and Azure AI development, pioneered enterprise AI integration through Copilot ecosystem. https://en.wikipedia.org/wiki/Satya_Nadella
		"nadella",

		// Allen Newell - Created Logic Theorist (first AI program), developed GPS and SOAR cognitive architectures, pioneered symbolic processing theory. https://en.wikipedia.org/wiki/Allen_Newell
		"newell",

		// Safiya Noble - Authored "Algorithms of Oppression", developed critical race framework for AI, pioneered research on search engine bias and algorithmic discrimination. https://en.wikipedia.org/wiki/Safiya_Noble
		"noble",

		// Chris Olah - Pioneered AI visualization techniques, developed network attribution methods, founded Anthropic's interpretability team, created Distill.pub. https://x.com/ch402
		"olah",

		// Chinasa T. Okolo - Develops AI applications for healthcare in low-resource settings, leads research on AI governance in Global South, advocates for inclusive AI development. https://en.wikipedia.org/wiki/Chinasa_T._Okolo
		"okolo",

		// Toby Ord - Oxford philosopher, author of "The Precipice", estimates high probability of AI-driven catastrophe. https://en.wikipedia.org/wiki/Toby_Ord
		"ord",

		// Niki Parmar - Co-author of "Attention Is All You Need", significant contributions to Transformer architecture and vision transformers. https://x.com/nikiparmar09
		"parmar",

		// Marc Raibert - Founder of Boston Dynamics, revolutionized dynamic robotics and legged locomotion. https://en.wikipedia.org/wiki/Marc_Raibert
		"raibert",

		// Gina Raimondo - US Secretary of Commerce, leading national AI policy and regulation. https://en.wikipedia.org/wiki/Gina_Raimondo
		"raimondo",

		// Sebastian Raschka - Created widely-used ML educational content and open-source tools. https://github.com/rasbt
		"raschka",

		// Nicholas Renotte - Creates comprehensive tutorials on implementing AI models. https://www.youtube.com/@NicholasRenotte
		"renotte",

		// Victor Riparbelli - CEO of Synthesia, pioneering AI video synthesis. https://x.com/vriparbelli
		"riparbelli",

		// Jonathan Ross - CEO of Groq, developing revolutionary AI accelerator chips and tensor processing units. https://x.com/JonathanRoss321
		"ross",

		// Wes Roth - Known for practical AI tutorials and implementation guides, particularly in LLMs and AI agents. https://x.com/JonathanRoss321
		"roth",

		// David Rumelhart - Co-developed backpropagation algorithm, pioneered parallel distributed processing. https://en.wikipedia.org/wiki/David_Rumelhart
		"rumelhart",

		// Cynthia Rudin - Pioneer in interpretable machine learning, advocate against black box models in high-stakes decisions. https://en.wikipedia.org/wiki/Cynthia_Rudin
		"rudin",

		// Daniela Rus - MIT CSAIL director, pioneering work in reconfigurable robotics and autonomous systems. https://en.wikipedia.org/wiki/Daniela_Rus
		"rus",

		// Arthur Samuel - Coined term "machine learning", created first self-learning programs. https://en.wikipedia.org/wiki/Arthur_Samuel_(computer_scientist)
		"samuel",

		// Grant Sanderson - Known for mathematical visualizations of ML concepts. https://www.youtube.com/c/3blue1brown
		"sanderson",

		// Marietje Schaake - International cyber policy leader, advocate for democratic AI governance. https://en.wikipedia.org/wiki/Marietje_Schaake
		"schaake",

		// Jürgen Schmidhuber - Early deep learning and meta-learning pioneer, supervised LSTM development. https://en.wikipedia.org/wiki/Jürgen_Schmidhuber
		"schmidhuber",

		// Terry Sejnowski - Pioneer in computational neuroscience and deep learning. https://en.wikipedia.org/wiki/Terry_Sejnowski
		"sejnowski",

		// Claude Shannon - Developed information theory fundamental to machine learning and neural networks. https://en.wikipedia.org/wiki/Claude_Shannon
		"shannon",

		// Noam Shazeer - Co-author of "Attention Is All You Need", invented the Mixture of Experts architecture, contributed to T5 and PaLM models. https://en.wikipedia.org/wiki/Noam_Shazeer
		"shazeer",

		// Divya Siddarth - Co-founder of Collective Intelligence Project, developing collective AI systems. https://en.wikipedia.org/wiki/Divya_Siddarth
		"siddarth",

		// Herbert Simon - Nobel laureate, developed bounded rationality theory, created Logic Theorist with Newell, pioneered heuristic problem solving in AI. https://en.wikipedia.org/wiki/Herbert_A._Simon
		"simon",

		// Aravind Srinivas - CEO of Perplexity AI, developed neural architecture search methods at Google Brain, pioneered efficient transformer training techniques. https://x.com/AravSrinivas
		"srinivas",

		// Josh Starmer - Created StatQuest, developed intuitive explanations of ML algorithms, pioneered visual approach to teaching statistical concepts in AI. https://www.youtube.com/c/joshstarmer
		"starmer",

		// YK Sugishita - Created CS Dojo, developed practical programming tutorials focusing on algorithms and ML fundamentals, pioneered accessible coding education. https://www.youtube.com/c/CSDojo
		"sugishita",

		// Mustafa Suleyman - Co-founded DeepMind and Inflection AI, developed AI governance frameworks, authored "The Coming Wave" on AI's societal impact. https://en.wikipedia.org/wiki/Mustafa_Suleyman
		"suleyman",

		// Ilya Sutskever - Previously OpenAI Chief Scientist, pioneered deep learning for sequence prediction, developed key insights in neural network optimization and scaling laws. https://en.wikipedia.org/wiki/Ilya_Sutskever
		"sutskever",

		// Jaan Tallinn - Skype co-founder, established Centre for the Study of Existential Risk, developed frameworks for AI x-risk assessment, major funder of AI alignment research. https://en.wikipedia.org/wiki/Jaan_Tallinn
		"tallinn",

		// Max Tegmark - Founded FLI and AI Safety Institute, developed mathematical universe hypothesis, authored influential AI safety frameworks in "Life 3.0". https://en.wikipedia.org/wiki/Max_Tegmark
		"tegmark",

		// Abhishek Thakur - First 4x Kaggle Grandmaster, developed automated ML pipeline frameworks, pioneered end-to-end approaches for competition-winning solutions. https://www.youtube.com/channel/UCBPRJjIWfyNG4X-CRbnv78A
		"thakur",

		// Rachel Thomas - Co-founded fast.ai, developed practical ML curriculum focusing on ethics and accessibility, pioneered "top-down" teaching approach for deep learning. https://x.com/math_rachel
		"thomas",

		// Helen Toner - CSET Director, shaped OpenAI's governance model, developed frameworks for assessing AI capabilities and strategic implications. https://en.wikipedia.org/wiki/Helen_Toner
		"toner",

		// Cari Tuna - Co-founded Open Philanthropy, pioneered cause prioritization in AI safety funding, developed frameworks for long-term AI risk assessment. https://en.wikipedia.org/wiki/Cari_Tuna
		"tuna",

		// Alan Turing - Developed Turing test, created foundations of computer science, pioneered machine intelligence concepts through mathematical formalization. https://en.wikipedia.org/wiki/Alan_Turing
		"turing",

		// Jakob Uszkoreit - Co-authored transformer architecture, led development of BERT at Google, pioneered neural machine translation architectures. https://x.com/kyosu
		"uszkoreit",

		// Ashwini Vaishnaw - India's IT Minister, developed national AI compute infrastructure plan, leads implementation of AI regulation framework. https://en.wikipedia.org/wiki/Ashwini_Vaishnaw
		"vaishnaw",

		// Leslie Valiant - Created PAC learning framework, developed computational learning theory, pioneered probably approximately correct approach to ML. https://en.wikipedia.org/wiki/Leslie_Valiant
		"valiant",

		// Shannon Vallor - Developed virtue ethics framework for AI, authored "AI & Moral Character", pioneered approaches for embedding ethics in AI development process. https://en.wikipedia.org/wiki/Shannon_Vallor
		"vallor",

		// Vladimir Vapnik - Created Statistical Learning Theory, developed Support Vector Machines (SVM), pioneered VC dimension theory for machine learning. https://en.wikipedia.org/wiki/Vladimir_Vapnik
		"vapnik",

		// Guillaume Verdon - "Beff Jezos" - founded Extropic, developed quantum tensor networks for AI, pioneered quantum-classical hybrid algorithms for machine learning. https://scholar.google.com/citations?user=DxFnepkAAAAJ
		"verdon",

		// Ashish Vaswani - Lead author of "Attention Is All You Need", developed transformer architecture at Google Brain, pioneered self-attention mechanisms. https://scholar.google.com/citations?user=bIU5MXUAAAAJ
		"vaswani",

		// C.C. Wei - TSMC CEO, developed 3nm/2nm processes for AI chips, leads development of specialized AI semiconductor manufacturing. https://en.wikipedia.org/wiki/C.C._Wei
		"wei",

		// Meredith Whittaker - Co-founded AI Now Institute, led Google's internal AI ethics protests, developed frameworks for algorithmic accountability. https://en.wikipedia.org/wiki/Meredith_Whittaker
		"whittaker",

		// Norbert Wiener - Founded cybernetics field, developed feedback control theory, authored "Cybernetics" connecting computation and control. https://en.wikipedia.org/wiki/Norbert_Wiener
		"wiener",

		// Thomas Wolf - Co-founded Hugging Face, developed transformers library, pioneered open-source model sharing and collaborative AI development. https://x.com/Thom_Wolf
		"wolf",

		// Steve Wozniak - Co-founded Apple, developed early personal computing architectures, advocates for ethical AI development and privacy rights. https://en.wikipedia.org/wiki/Steve_Wozniak
		"wozniak",

		// Wang Xiaochuan - Founded Baichuan AI, developed open-source LLMs in China, pioneered bilingual model architectures and training techniques. https://en.wikipedia.org/wiki/Wang_Xiaochuan
		"xiaochuan",

		// Tom Yeh - Pioneered "AI by hand" approach, developed novel techniques for human-AI collaboration in art and design, leads research on interpretable AI visualization. https://x.com/proftomyeh
		"yeh",

		// Eliezer Yudkowsky - Founded MIRI, developed coherent extrapolated volition framework, authored "Rationality: A-Z", pioneered early work on AI alignment theory. https://en.wikipedia.org/wiki/Eliezer_Yudkowsky
		"yudkowsky",

		// Károly Zsolnai-Fehér - Created Two Minute Papers, developed novel techniques for explaining complex AI research, pioneered accessible ML education through visual demonstrations. https://www.youtube.com/c/K%C3%A1rolyZsolnai
		"zsolnai",

		// Mark Zuckerberg - Meta CEO, leads development of large language models like LLaMA, pioneered open-source AI model releases, drives AI integration in social platforms. https://en.wikipedia.org/wiki/Mark_Zuckerberg
		"zuckerberg",

		// David Silver - DeepMind research scientist who led AlphaGo project, pioneering work in reinforcement learning and game-playing AI. https://en.wikipedia.org/wiki/David_Silver_(computer_scientist)
		"silver",

		// Lisa Su - AMD CEO leading development of AI accelerators and neural engine processors, pioneered chiplet architecture for AI. https://en.wikipedia.org/wiki/Lisa_Su
		"su",

		// Andrew Feldman - Founded Cerebras Systems, developed world's largest AI chip and wafer-scale engine technology. https://x.com/andrewdfeldman
		"feldman",

		// Yanjun Ma - Chief Scientist at Alibaba DAMO Academy, pioneering work in large-scale AI systems and multilingual models. https://scholar.google.com/citations?user=tVWZjPwAAAAJ
		"ma",

		// Terry Winograd - Created SHRDLU natural language system in 1970, pioneer in AI natural language processing. https://en.wikipedia.org/wiki/Terry_Winograd
		"winograd",

		// Finale Doshi-Velez - Harvard professor pioneering interpretable machine learning and trustworthy AI systems. https://en.wikipedia.org/wiki/Finale_Doshi-Velez
		"doshivelez",
	}
)

// Generate generates a random name from the list of adjectives and surnames in this package
// formatted as "adjective_surname". For example 'focused_turing'. If retry is non-zero, a random
// integer between 0 and 10 will be added to the end of the name, e.g `focused_turing3`
func Generate(retry int) string {
begin:
	name := left[rand.Intn(len(left))] + "_" + right[rand.Intn(len(right))] //nolint:gosec // G404: Use of weak random number generator (math/rand instead of crypto/rand)
	if name == "honest_altman" /* Sam Altman is not honest */ {
		goto begin
	}

	if retry > 0 {
		name += strconv.Itoa(rand.Intn(10)) //nolint:gosec // G404: Use of weak random number generator (math/rand instead of crypto/rand)
	}
	return name
}
