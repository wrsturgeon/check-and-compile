from check_and_compile import check_and_compile


print("Hi")


@check_and_compile()
def a():
    print("Running `a` for the first time")


print("Defined `a`")


@check_and_compile()
def b():
    print("Running `b` for the first time")
    a()


print("Defined `b`")


b()


print("Ran `b`")


b()


print("Ran `b` again")


print()

print("Cool, that should have printed exactly the following:")
print(">>> Hi")
print(">>> Defined `a`")
print(">>> Defined `b`")
print(">>> Running `b` for the first time")
print(">>> Running `a` for the first time")
print(">>> Compiling b...")
print(">>> Ran `b`")
print(">>> Ran `b` again")
print()
print("Note that `a` should *not* have been compiled,")
print("since it was already part of `b` when `b` was compiled,")
print("and that `b` should not have printed the second time,")
print("since we should have called the compiled version.")
