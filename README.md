# agentnames
Package agentnames generates random names for agents based on famous AI/ML folks.


<!-- Code generated by gomarkdoc. DO NOT EDIT -->

# agentnames

```go
import "github.com/agentstation/agentnames"
```

Package agentnames generates random names for agents based on famous AI/ML folks.

## Index

- [func Generate\(retry int\) string](<#Generate>)


<a name="Generate"></a>
## func [Generate](<https://github.com/agentstation/agentnames/blob/master/agentnames.go#L838>)

```go
func Generate(retry int) string
```

Generate generates a random name from the list of adjectives and surnames in this package formatted as "adjective\_surname". For example 'focused\_turing'. If retry is non\-zero, a random integer between 0 and 10 will be added to the end of the name, e.g \`focused\_turing3\`

Generated by [gomarkdoc](<https://github.com/princjef/gomarkdoc>)
