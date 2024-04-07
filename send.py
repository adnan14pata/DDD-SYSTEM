from twilio.rest import Client

account_sid = 'ACd7bd4788a5306c92c0c0b852a50f0609'
auth_token = '32f230b53385bc7f21ac2df3a7f33cb4'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from='+12513062866',
  body='alert!!! possible accident  ',
  to='+919834121604'
)

print(message.sid)