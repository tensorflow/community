# C++ Coding Style

Tensorflow follows [Google C++ style](https://google.github.io/styleguide/cppguide.html),
with a few additions.

## Status

Functions which can produce an error should return a `tensorflow::Status`. To propagate an
error status, use the `TF_RETURN_IF_ERROR` macro.

```
TF_RETURN_IF_ERROR(f());
```

## StatusOr<T>

`StatusOr<T>` is the union of a `Status` object and a `T` object. It offers a way to use
return values instead of output parameters for functions which may fail.

For example, consider the code:

```
Output out;
Status s = foo(&out);
if (!s.ok()) {
  return s;
}
out.DoSomething();
```

With `StatusOr<T>`, we can write this as

```
StatusOr<Output> result = foo();
if (!result.ok()) {
  return result.status();
}
result->DoSomething();
```

**Pros:**

Return values are
[easier to reason about](https://google.github.io/styleguide/cppguide.html#Output_Parameters)
than output parameters.

The types returned through `StatusOr<T>` don't need to support empty states. To return a type
as an output parameter, we must either use a `unique_ptr<T>` or support an empty state for the
type so that we can initialize the type before passing it as an output parameter. `StatusOr<T>`
reduces the number of objects we have in an "uninitialized" state.

**Cons:**

`StatusOr<T>` adds complexity. It raises questions about what happens when `T` is null and
how `StatusOr<T>` behaves during moves and copies. `StatusOr<T>` also generally comes with
macros such as `ASSIGN_OR_RETURN`, which add additional complexity.

The current Tensorflow codebase exclusively uses `Status` instead of `StatusOr<T>`, so
switching over would require a significant amount of work.

**Decision:**

Tensorflow foregoes the use of `StatusOr<>` because it doesn't add enough value to justify
additional complexity.
