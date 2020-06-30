# cognimaker


## Dev notes

### mypy annotations

- @dataclasses create spurious warnings when combined with ABC or Callable members;
  these warnings are ignored with comments
- sometimes using an empty list or dict literal such as `[]` will cause mypy to
  give a 'Need type annotation' error, so sometimes variables are annotated with
  e.g. `List[Any]` even though it doesn't convey much information