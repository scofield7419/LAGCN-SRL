
#!/bin/sh

set -e

root=$(cd "$(dirname "$0")"; pwd)
pkg=se.liu.ida.nlp.sdp.toolkit.tools.

java -cp $root/sdp.jar $pkg$@
